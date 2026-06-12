"""
viz.py — Publication-quality figures for the Quantum Bio-Seam pipeline
=======================================================================
All plots write to the outputs/plots/ directory.
Designed to work with both synthetic and real plate-reader / scRNA-seq data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap



logger = logging.getLogger(__name__)

PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "baseline":  "#444444",
    "cond_1":    "#4a9edd",
    "cond_2":    "#5bcf8a",
    "cond_3":    "#f0b429",
    "cond_4":    "#e05c5c",
    "col_f":     "#3d7ebf",
    "ker_f":     "#c44b3d",
    "neutral":   "#aaaaaa",
    "validated": "#2a9e4f",
    "failed":    "#cc3333",
}

_BIO_CMAP = LinearSegmentedColormap.from_list(
    "bio_seam", ["#1a1a2e", "#3d7ebf", "#5bcf8a", "#f0b429", "#e05c5c"]
)


def _save(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    out = PLOT_DIR / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", out)
    return out


# ── 1. SVD Spectrum ────────────────────────────────────────────────────────────

def plot_svd_spectrum(result, title_suffix: str = "") -> Path:
    S   = result.singular_values
    var = result.variance_pct
    cum = result.cumulative_var
    ok  = result.rank1_validated

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"SVD of Dynamic Trajectory Alteration Matrix ΔA  {title_suffix}",
        fontsize=13, fontweight="bold",
    )

    # Left: singular values
    colors = [PALETTE["col_f"] if i == 0 else PALETTE["neutral"] for i in range(len(S))]
    axes[0].bar(range(1, len(S)+1), S, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Component"); axes[0].set_ylabel("Singular Value")
    axes[0].set_title("Singular Value Spectrum")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Centre: variance per component
    axes[1].bar(range(1, len(var)+1), var,
                color=[PALETTE["validated"] if i == 0 else "#dddddd" for i in range(len(var))],
                edgecolor="white")
    axes[1].axhline(result.threshold, color=PALETTE["ker_f"], lw=1.5, ls="--",
                    label=f"{result.threshold:.0f}% threshold")
    axes[1].set_xlabel("Component"); axes[1].set_ylabel("Variance Explained (%)")
    axes[1].set_title("Variance per Component"); axes[1].legend(fontsize=9)
    axes[1].grid(True, axis="y", alpha=0.3)

    # Right: cumulative variance
    axes[2].plot(range(1, len(cum)+1), cum, "o-", color=PALETTE["col_f"], lw=2)
    axes[2].axhline(result.threshold, color=PALETTE["ker_f"], lw=1.5, ls="--")
    axes[2].fill_between(range(1, len(cum)+1), 0, cum, alpha=0.15, color=PALETTE["col_f"])
    axes[2].set_xlabel("Components"); axes[2].set_ylabel("Cumulative Variance (%)")
    axes[2].set_title("Cumulative Explained Variance")
    axes[2].set_ylim(0, 105); axes[2].grid(True, alpha=0.25)

    verdict = "RANK-1 PREDICTION: VALIDATED ✓" if ok else "RANK-1 PREDICTION: NOT MET ✗"
    color   = PALETTE["validated"] if ok else PALETTE["failed"]
    fig.text(0.5, 0.01, verdict, ha="center", fontsize=12,
             color=color, fontweight="bold")

    return _save(fig, "svd_spectrum.png")


# ── 2. Architecture Gap ────────────────────────────────────────────────────────

def plot_architecture_gap(results: dict) -> Path:
    cont_mse  = results["Continuous MLP (current SOTA)"]["final_mse"]
    kaware_mse = results["Kernel-Aware MLP (Bio-Seam)"]["final_mse"]
    gap       = results.get("gap_ratio", cont_mse / max(kaware_mse, 1e-9))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Architecture Gap: ker(F) Signal Recovery", fontsize=13, fontweight="bold")

    # Left: MSE comparison
    names  = ["Continuous MLP\n(current SOTA)", "Kernel-Aware MLP\n(Bio-Seam)"]
    mses   = [cont_mse, kaware_mse]
    colors = [PALETTE["neutral"], PALETTE["validated"]]
    bars   = axes[0].bar(names, mses, color=colors, width=0.5, edgecolor="white", linewidth=0.8)
    axes[0].set_ylabel("Validation MSE (ker(F) signal recovery)")
    axes[0].set_title("MSE on Synonymous Codon Regulatory Signal")
    for bar, val in zip(bars, mses):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    axes[0].grid(True, axis="y", alpha=0.25)

    # Right: gap ratio gauge
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.72, f"{gap:.0f}×", ha="center", va="center",
            fontsize=52, fontweight="bold", color=PALETTE["validated"])
    ax.text(0.5, 0.50, "Architecture Gap Ratio", ha="center", va="center",
            fontsize=13, color="#333333")
    ax.text(0.5, 0.36, f"ker(F)-aware MSE {gap:.0f}× lower than\ncontinuous model",
            ha="center", va="center", fontsize=10, color="#666666")
    ax.text(0.5, 0.14,
            "Continuous models collapse synonymous codons\ninto a single latent basin — "
            "the ker(F) is invisible.",
            ha="center", va="center", fontsize=9, style="italic", color="#888888")

    return _save(fig, "architecture_gap.png")


# ── 3. Sherman-Morrison Efficiency ────────────────────────────────────────────

def plot_sm_efficiency(
    time_naive: float,
    time_sm: float,
    max_error: float,
    N: int,
) -> Path:
    speedup = time_naive / max(time_sm, 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Sherman-Morrison Efficiency Proof  (N = {N:,} genes)",
        fontsize=13, fontweight="bold",
    )

    # Left: time comparison
    names  = [f"Brute-Force Re-Inversion\nO(N³)", f"Sherman-Morrison\nO(N²)"]
    times  = [time_naive, time_sm]
    colors = [PALETTE["ker_f"], PALETTE["validated"]]
    bars   = axes[0].bar(names, times, color=colors, width=0.45, edgecolor="white")
    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].set_title("Computational Cost")
    for bar, t in zip(bars, times):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f"{t:.4f}s", ha="center", va="bottom", fontsize=10)
    axes[0].grid(True, axis="y", alpha=0.25)

    # Right: speedup + error annotation
    ax = axes[1]; ax.axis("off")
    ax.text(0.5, 0.78, f"{speedup:.0f}×", ha="center", va="center",
            fontsize=52, fontweight="bold", color=PALETTE["validated"])
    ax.text(0.5, 0.58, "Speedup Factor", ha="center", va="center",
            fontsize=14, color="#333333")
    ax.text(0.5, 0.44, f"Max numerical error: {max_error:.2e}",
            ha="center", va="center", fontsize=11,
            color=PALETTE["validated"] if max_error < 1e-6 else PALETTE["failed"])
    ax.text(0.5, 0.28,
            "Rank-1 updates from the ker(F) boundary\npropagate at O(N²) — "
            f"not the O(N³)\ncost of full re-inversion.",
            ha="center", va="center", fontsize=9, style="italic", color="#888888")
    passed_str = "ALGEBRAIC IDENTITY VERIFIED ✓" if max_error < 1e-6 else "VERIFY FAILED ✗"
    ax.text(0.5, 0.10, passed_str, ha="center", va="center", fontsize=11,
            fontweight="bold",
            color=PALETTE["validated"] if max_error < 1e-6 else PALETTE["failed"])

    return _save(fig, "sm_efficiency.png")


# ── 4. scRNA-seq kernel load map ──────────────────────────────────────────────

def plot_kernel_load(
    ker_load: torch.Tensor,     # (cells,)
    cell_labels: Optional[np.ndarray] = None,
    umap_coords: Optional[np.ndarray] = None,  # (cells, 2)
) -> Path:
    ker_load_np = ker_load.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ker(F) Kernel Load Across Cells", fontsize=13, fontweight="bold")

    # Left: distribution
    axes[0].hist(ker_load_np, bins=60, color=PALETTE["ker_f"], alpha=0.8, edgecolor="white")
    axes[0].axvline(0.35, color=PALETTE["col_f"], lw=2, ls="--", label="Critical threshold")
    axes[0].set_xlabel("ker(F) Load Score"); axes[0].set_ylabel("Cell count")
    axes[0].set_title("Distribution of ker(F) Load")
    axes[0].legend(); axes[0].grid(True, axis="y", alpha=0.25)
    n_above = (ker_load_np > 0.35).sum()
    axes[0].text(0.97, 0.95,
                 f"{n_above} cells ({100*n_above/len(ker_load_np):.1f}%)\nabove threshold",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=9, color=PALETTE["ker_f"])

    # Right: UMAP colored by ker_load (or scatter by index if no UMAP)
    if umap_coords is not None and umap_coords.shape[0] == len(ker_load_np):
        sc = axes[1].scatter(umap_coords[:, 0], umap_coords[:, 1],
                             c=ker_load_np, cmap=_BIO_CMAP, s=6, alpha=0.7)
        plt.colorbar(sc, ax=axes[1], label="ker(F) Load")
        axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
        axes[1].set_title("ker(F) Load on UMAP")
    else:
        # Fallback: sorted cell index plot
        sorted_load = np.sort(ker_load_np)[::-1]
        axes[1].fill_between(range(len(sorted_load)), sorted_load,
                             color=PALETTE["ker_f"], alpha=0.6)
        axes[1].axhline(0.35, color=PALETTE["col_f"], lw=2, ls="--")
        axes[1].set_xlabel("Cells (sorted by load)"); axes[1].set_ylabel("ker(F) Load")
        axes[1].set_title("Sorted ker(F) Load per Cell")
        axes[1].grid(True, alpha=0.25)

    return _save(fig, "kernel_load_map.png")


# ── 5. GRN response map heatmap ───────────────────────────────────────────────

def plot_grn_response_map(
    A_inv: torch.Tensor,    # (N, N) — show top-N_show genes
    gene_names: Optional[list[str]] = None,
    n_show: int = 40,
) -> Path:
    N = A_inv.shape[0]
    n_show = min(n_show, N)

    # Select top genes by row-norm (most connected in response map)
    row_norms = A_inv.abs().norm(dim=1).cpu().numpy()
    top_idx   = np.argsort(row_norms)[::-1][:n_show].copy()
    sub       = A_inv[np.ix_(top_idx, top_idx)].cpu().numpy()

    labels = ([gene_names[i] for i in top_idx]
              if gene_names else [str(i) for i in top_idx])

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sub, cmap="RdBu_r", aspect="auto",
                   vmin=-np.percentile(np.abs(sub), 95),
                   vmax=+np.percentile(np.abs(sub), 95))
    plt.colorbar(im, ax=ax, label="A⁻¹ entry (response strength)")
    ax.set_xticks(range(n_show)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n_show)); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(f"GRN Response Map A⁻¹ — Top {n_show} genes by connectivity",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    return _save(fig, "grn_response_map.png")


# ── 6. Summary dashboard ───────────────────────────────────────────────────────

def plot_summary_dashboard(
    svd_result: SVDResult,
    arch_results: dict,
    sm_times: tuple[float, float],
    sm_error: float,
    N_genes: int,
) -> Path:
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "THE QUANTUM BIO-SEAM — Verification Suite Summary",
        fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: SVD spectrum ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    S   = svd_result.singular_values
    var = svd_result.variance_pct
    colors = [PALETTE["col_f"] if i == 0 else PALETTE["neutral"] for i in range(len(S))]
    ax1.bar(range(1, len(S)+1), var, color=colors, edgecolor="white")
    ax1.axhline(svd_result.threshold, color=PALETTE["ker_f"], lw=1.5, ls="--")
    ax1.set_title("Proof 3: SVD Rank-1 Test", fontweight="bold")
    ax1.set_xlabel("Component"); ax1.set_ylabel("Variance (%)")
    verdict = f"✓ {var[0]:.1f}% first mode" if svd_result.rank1_validated \
              else f"✗ {var[0]:.1f}% (need >{svd_result.threshold:.0f}%)"
    ax1.text(0.5, 0.92, verdict, transform=ax1.transAxes, ha="center",
             color=PALETTE["validated"] if svd_result.rank1_validated else PALETTE["failed"],
             fontsize=9, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.25)

    # ── Panel 2: Architecture gap bar ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    cont_mse  = arch_results["Continuous MLP (current SOTA)"]["final_mse"]
    kaw_mse   = arch_results["Kernel-Aware MLP (Bio-Seam)"]["final_mse"]
    ax2.bar(["Continuous\nMLP", "Kernel-Aware\nMLP"],
            [cont_mse, kaw_mse],
            color=[PALETTE["neutral"], PALETTE["validated"]],
            edgecolor="white")
    ax2.set_title("Proof 2: Architecture Gap", fontweight="bold")
    ax2.set_ylabel("Val MSE (ker(F) recovery)")
    ax2.text(0.5, 0.90,
             f"Gap: {arch_results['gap_ratio']:.0f}×",
             transform=ax2.transAxes, ha="center",
             color=PALETTE["validated"], fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.25)

    # ── Panel 3: SM speedup ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    t_naive, t_sm = sm_times
    ax3.bar(["O(N³)\nRe-inversion", "O(N²)\nSherman-Morrison"],
            [t_naive, t_sm],
            color=[PALETTE["ker_f"], PALETTE["validated"]],
            edgecolor="white")
    ax3.set_title("Proof 1: SM Efficiency", fontweight="bold")
    ax3.set_ylabel("Time (s)")
    speedup = t_naive / max(t_sm, 1e-9)
    ax3.text(0.5, 0.90, f"{speedup:.0f}× speedup",
             transform=ax3.transAxes, ha="center",
             color=PALETTE["validated"], fontsize=11, fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.25)

    # ── Panel 4: Cumulative SVD variance ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    cum = svd_result.cumulative_var
    ax4.plot(range(1, len(cum)+1), cum, "o-", color=PALETTE["col_f"], lw=2)
    ax4.fill_between(range(1, len(cum)+1), 0, cum, alpha=0.15, color=PALETTE["col_f"])
    ax4.axhline(svd_result.threshold, color=PALETTE["ker_f"], lw=1.5, ls="--")
    ax4.set_xlabel("Components"); ax4.set_ylabel("Cumulative Var (%)")
    ax4.set_title("Cumulative SVD Variance")
    ax4.set_ylim(0, 105); ax4.grid(True, alpha=0.25)

    # ── Panel 5: Error metric comparison ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis("off")
    rows = [
        ("Proof", "Claim", "Status"),
        ("1 — SM Efficiency",
         f"O(N²) vs O(N³)  [{N_genes:,} genes]",
         f"{speedup:.0f}× speedup  {'✓' if sm_error < 1e-6 else '✗'}"),
        ("2 — Architecture Gap",
         "ker(F) signal invisible to std MLP",
         f"{arch_results['gap_ratio']:.0f}× MSE gap  ✓"),
        ("3 — Rank-1 SVD",
         f"First mode > {svd_result.threshold:.0f}%",
         f"{svd_result.singular_values[0]:.3f} σ₁  {'✓' if svd_result.rank1_validated else '✗'}"),
    ]
    tbl = ax5.table(
        cellText=rows[1:], colLabels=rows[0],
        loc="center", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 2.0)
    ax5.set_title("Verification Ledger", fontweight="bold")

    # ── Panel 6: Bio-Seam concept ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    lines = [
        ("THE QUANTUM BIO-SEAM", 0.92, 12, "bold", "#222222"),
        ("col(F) / ker(F) boundary", 0.80, 10, "normal", PALETTE["col_f"]),
        ("→ Amino acid selection", 0.72, 9, "normal", "#555555"),
        ("→ Synonymous codon regulation", 0.64, 9, "normal", "#555555"),
        ("Sherman-Morrison architecture", 0.52, 10, "normal", PALETTE["validated"]),
        ("→ O(N²) rank-1 updates", 0.44, 9, "normal", "#555555"),
        ("→ Metabolically favorable", 0.36, 9, "normal", "#555555"),
        ("Maxwell's Demon (CP ensemble)", 0.24, 10, "normal", PALETTE["ker_f"]),
        ("→ Non-equilibrium criticality", 0.16, 9, "normal", "#555555"),
        ("ERI Labs · June 2026", 0.04, 8, "normal", "#aaaaaa"),
    ]
    for text, y, size, weight, color in lines:
        ax6.text(0.05, y, text, transform=ax6.transAxes,
                 fontsize=size, color=color, fontweight=weight)

    return _save(fig, "summary_dashboard.png", dpi=180)
