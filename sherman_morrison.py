"""
sherman_morrison.py — Sherman-Morrison GRN Engine
==================================================
Core matrix-math module for the Quantum Bio-Seam pipeline.

Implements:
  1. GRNEngine          — builds Gene Regulatory Network from expression data,
                          applies and accumulates rank-1 Sherman-Morrison updates
  2. RankOneDetector    — SVD-based test for whether a perturbation response
                          is consistent with a rank-1 update (>90% first mode)
  3. KernelLoadEstimator— measures ker(F) tRNA bottleneck as a scalar signal

All operations are batched and GPU-aware.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sym_regularize(A: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Symmetrize and add ridge regularization for invertibility."""
    A = (A + A.T) / 2.0
    A = A + eps * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    return A


def _safe_inv(A: torch.Tensor) -> torch.Tensor:
    """Invert with fallback to pseudo-inverse on singular matrices."""
    try:
        return torch.linalg.inv(A)
    except torch.linalg.LinAlgError:
        logger.warning("Matrix singular; using pseudo-inverse.")
        return torch.linalg.pinv(A)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GRN Engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UpdateRecord:
    """Record of a single rank-1 Sherman-Morrison update."""
    update_id:   int
    u:           torch.Tensor   # (N,)
    v:           torch.Tensor   # (N,)
    denom:       float
    max_delta:   float          # max |ΔA⁻¹| entry
    time_s:      float


class GRNEngine:
    """
    Gene Regulatory Network Engine.

    Workflow
    --------
    1. grn = GRNEngine.from_expression(X)  — builds A from scRNA-seq data
    2. grn.invert()                        — compute A⁻¹ (one-time cost)
    3. grn.update(u, v)                    — apply rank-1 SM update
    4. grn.update(u, v)                    — accumulate further updates
    5. grn.response_map                    — current (A + ΣuvT)⁻¹

    All tensors live on the same device as the input expression matrix.
    """

    def __init__(self, A: torch.Tensor, gene_names: Optional[list[str]] = None):
        assert A.ndim == 2 and A.shape[0] == A.shape[1], "A must be square"
        self.N = A.shape[0]
        self.device = A.device
        self.dtype  = A.dtype
        self.gene_names = gene_names or [f"gene_{i}" for i in range(self.N)]

        self.A      = A
        self.A_inv: Optional[torch.Tensor] = None
        self._updates: list[UpdateRecord] = []
        self._update_counter = 0

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_expression(
        cls,
        X: torch.Tensor,                   # (cells, genes)
        gene_names: Optional[list[str]] = None,
        reg_eps: float = 1e-3,
        method: str = "pearson",           # "pearson" | "covariance" | "identity"
        max_genes: int = 5000,
    ) -> "GRNEngine":
        """
        Build GRN adjacency matrix A from expression matrix X.

        method="pearson"    — Pearson correlation (standard; scale-invariant)
        method="covariance" — sample covariance (preserves magnitude)
        method="identity"   — debug/benchmark mode
        """
        cells, genes = X.shape
        logger.info(
            "Building GRN: %d cells × %d genes (method=%s)", cells, genes, method
        )

        # Sub-select genes if too many (memory guard)
        if genes > max_genes:
            logger.warning(
                "Gene count %d > max_genes %d; selecting top-%d by variance.",
                genes, max_genes, max_genes,
            )
            var = X.var(dim=0)
            top_idx = torch.topk(var, max_genes).indices
            X = X[:, top_idx]
            genes = max_genes
            if gene_names:
                gene_names = [gene_names[i] for i in top_idx.cpu().tolist()]

        if method == "identity":
            A = torch.eye(genes, dtype=X.dtype, device=X.device)
        elif method == "covariance":
            Xc = X - X.mean(dim=0, keepdim=True)
            A  = (Xc.T @ Xc) / (cells - 1)
        else:  # pearson
            Xc  = X - X.mean(dim=0, keepdim=True)
            std = Xc.std(dim=0, keepdim=True).clamp(min=1e-8)
            Xn  = Xc / std
            A   = (Xn.T @ Xn) / (cells - 1)

        A = _sym_regularize(A, eps=reg_eps * genes)
        logger.info("GRN matrix built: %d × %d", genes, genes)
        return cls(A, gene_names=gene_names)

    # ── Inversion ────────────────────────────────────────────────────────────

    def invert(self, force: bool = False) -> "GRNEngine":
        """Compute A⁻¹. Cached — only runs once unless force=True."""
        if self.A_inv is not None and not force:
            return self
        t0 = time.perf_counter()
        logger.info("Inverting %d×%d GRN (O(N³) — one-time cost)…", self.N, self.N)
        self.A_inv = _safe_inv(self.A)
        dt = time.perf_counter() - t0
        logger.info("Inversion complete: %.3f s", dt)
        return self

    # ── Sherman-Morrison rank-1 update ────────────────────────────────────────

    def update(
        self,
        u: torch.Tensor,   # (N,) or (N, 1) — perturbation signal vector
        v: torch.Tensor,   # (N,) or (N, 1) — propagation vector
        label: Optional[str] = None,
    ) -> UpdateRecord:
        """
        Apply one rank-1 Sherman-Morrison update in-place to A_inv.

        (A + uvᵀ)⁻¹ = A⁻¹ − (A⁻¹u)(vᵀA⁻¹) / (1 + vᵀA⁻¹u)

        Cost: O(N²)  — two matrix-vector products.
        """
        if self.A_inv is None:
            raise RuntimeError("Call .invert() before .update()")

        u = u.to(self.device, self.dtype).reshape(-1, 1)   # (N, 1)
        v = v.to(self.device, self.dtype).reshape(-1, 1)   # (N, 1)

        t0 = time.perf_counter()

        A_inv_u   = self.A_inv @ u                         # (N, 1)  O(N²)
        vT_A_inv  = v.T @ self.A_inv                       # (1, N)  O(N²)
        scalar    = 1.0 + (v.T @ A_inv_u).item()          # scalar

        if abs(scalar) < 1e-12:
            logger.warning("SM denom ≈ 0 (near-degenerate update); skipping.")
            return None

        correction = (A_inv_u @ vT_A_inv) / scalar        # (N, N)  O(N²)

        old_inv = self.A_inv.clone()
        self.A_inv = self.A_inv - correction

        max_delta = (self.A_inv - old_inv).abs().max().item()
        dt = time.perf_counter() - t0

        self._update_counter += 1
        rec = UpdateRecord(
            update_id = self._update_counter,
            u         = u.squeeze().cpu(),
            v         = v.squeeze().cpu(),
            denom     = scalar,
            max_delta = max_delta,
            time_s    = dt,
        )
        self._updates.append(rec)

        logger.debug(
            "SM update #%d: denom=%.4f, max_Δ=%.3e, %.4fs  [%s]",
            self._update_counter, scalar, max_delta, dt, label or ""
        )
        return rec

    # ── Batched updates (from scRNA perturbation conditions) ─────────────────

    def batch_update_from_conditions(
        self,
        condition_means: torch.Tensor,   # (C, N) — mean expression per condition
        baseline_idx: int = 0,
    ) -> list[UpdateRecord]:
        """
        For each non-baseline condition, compute the perturbation vector
        relative to baseline and apply as a rank-1 SM update.

        u = Δμ (expression shift)
        v = Δμ / ||Δμ|| (normalized direction — encodes propagation)
        """
        records = []
        baseline = condition_means[baseline_idx]  # (N,)
        for i, cond_mean in enumerate(condition_means):
            if i == baseline_idx:
                continue
            delta = cond_mean - baseline          # (N,)
            u = delta
            v = F.normalize(delta, dim=0)
            rec = self.update(u, v, label=f"condition_{i}")
            if rec:
                records.append(rec)
        return records

    @property
    def response_map(self) -> torch.Tensor:
        """Current (A + Σ uvᵀ)⁻¹ — the cell's global response map."""
        return self.A_inv

    @property
    def n_updates(self) -> int:
        return self._update_counter


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rank-1 Detector (SVD validation)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SVDResult:
    singular_values:   np.ndarray
    variance_pct:      np.ndarray
    cumulative_var:    np.ndarray
    first_mode_pct:    float
    rank1_validated:   bool           # first mode > threshold
    threshold:         float
    n_conditions:      int
    n_timepoints:      int
    pearson_r:         Optional[float] = None
    pearson_p:         Optional[float] = None


class RankOneDetector:
    """
    Runs SVD on the Dynamic Trajectory Alteration Matrix (ΔA) and
    tests the Sherman-Morrison rank-1 prediction.

    ΔA[i, t] = trajectory(condition_i, t) − trajectory(baseline, t)

    Bio-Seam prediction: S₀² / ΣSₖ² > threshold  (default 0.90)
    """

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

    def build_delta_matrix(
        self,
        trajectories: torch.Tensor,   # (C, T) — C conditions, T time points
        baseline_idx: int = 0,
    ) -> torch.Tensor:
        """
        Subtract baseline trajectory from each condition.
        Returns ΔA of shape (C-1, T).
        """
        baseline = trajectories[baseline_idx:baseline_idx+1, :]  # (1, T)
        mask = torch.ones(trajectories.shape[0], dtype=torch.bool)
        mask[baseline_idx] = False
        delta = trajectories[mask] - baseline      # (C-1, T)
        return delta

    def run(
        self,
        trajectories: torch.Tensor,   # (C, T)
        baseline_idx: int = 0,
        aux_signal: Optional[torch.Tensor] = None,  # (C-1,) for Pearson
    ) -> SVDResult:
        """
        Full SVD validation pipeline.
        """
        delta = self.build_delta_matrix(trajectories, baseline_idx)
        C1, T = delta.shape

        # SVD on CPU numpy (torch SVD equivalently fine)
        DA = delta.cpu().float().numpy()
        U, S, Vt = np.linalg.svd(DA, full_matrices=False)

        var_pct = (S ** 2) / (S ** 2).sum() * 100.0
        cum_var = np.cumsum(var_pct)

        rank1_ok = (var_pct[0] / 100.0) > self.threshold

        # Optional Pearson: first singular mode score vs aux signal
        r, p = None, None
        if aux_signal is not None:
            from scipy.stats import pearsonr
            scores = DA @ Vt[0]   # projection onto first right singular vector
            aux    = aux_signal.cpu().numpy()
            if len(aux) == len(scores):
                r, p = pearsonr(scores, aux)

        result = SVDResult(
            singular_values = S,
            variance_pct    = var_pct,
            cumulative_var  = cum_var,
            first_mode_pct  = float(var_pct[0]),
            rank1_validated = rank1_ok,
            threshold       = self.threshold * 100.0,
            n_conditions    = C1,
            n_timepoints    = T,
            pearson_r       = r,
            pearson_p       = p,
        )

        logger.info(
            "SVD: first mode %.1f%%  (threshold >%.0f%%)  → %s",
            var_pct[0], self.threshold * 100.0,
            "VALIDATED ✓" if rank1_ok else "NOT VALIDATED ✗",
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. ker(F) Load Estimator
# ─────────────────────────────────────────────────────────────────────────────

class KernelLoadEstimator:
    """
    Estimates the ker(F) tRNA bottleneck load for each cell from:
      - X_ker  : expression of ker(F)-partitioned genes  (cells × K)
      - X_col  : expression of col(F)-partitioned genes  (cells × C)

    Metrics
    -------
    ker_load     — ratio of ker mean expression to col mean expression
                   (high = tRNA bottleneck active)
    ratiometric  — GFP/mCherry analog; normalized per-cell
    bottleneck   — cells where ker_load exceeds the critical threshold
    """

    def __init__(self, critical_threshold: float = 0.35):
        self.critical_threshold = critical_threshold

    def compute(
        self,
        X_ker: torch.Tensor,   # (cells, K)
        X_col: torch.Tensor,   # (cells, C)
        eps: float = 1e-6,
    ) -> dict[str, torch.Tensor]:
        ker_mean = X_ker.mean(dim=1)           # (cells,)
        col_mean = X_col.mean(dim=1).clamp(min=eps)

        ker_load    = ker_mean / col_mean      # ratiometric load
        bottleneck  = ker_load > self.critical_threshold

        # Per-cell normalized ratio (analog of GFP/mCherry ratiometric read)
        ratio_norm = (ker_mean - ker_mean.mean()) / (ker_mean.std() + eps)

        return {
            "ker_load":         ker_load,
            "ratiometric":      ratio_norm,
            "bottleneck_mask":  bottleneck,
            "n_bottleneck":     bottleneck.sum().item(),
            "pct_bottleneck":   bottleneck.float().mean().item() * 100.0,
        }
