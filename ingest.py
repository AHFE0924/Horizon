"""
ingest.py — scRNA-seq Matrix Ingestion & Preprocessing
=======================================================
Accepts: 10x Genomics MEX (matrix.mtx.gz + barcodes + features),
         AnnData (.h5ad), plain .h5, or a dense CSV/TSV.

Outputs a normalized, log1p-scaled torch.Tensor (cells × genes)
plus a CodonMatrix object that partitions genes into col(F) / ker(F)
bins for downstream Bio-Seam analysis.

All heavy lifting stays on GPU when available.
"""

from __future__ import annotations

import gzip
import logging
import os
from pathlib import Path
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

logger = logging.getLogger(__name__)

# dtype / device helpers 

def _get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_dense_numpy(X) -> np.ndarray:
    """Coerce sparse or ndarray to float32 dense numpy array."""
    if sp.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


# Loaders 
def load_10x_mex(folder: Union[str, Path]) -> ad.AnnData:
    """
    Load 10x MEX directory.
    Expects: matrix.mtx[.gz], barcodes.tsv[.gz], features.tsv[.gz] (or genes.tsv).
    """
    folder = Path(folder)
    mtx_path = next(folder.glob("matrix.mtx*"))
    bc_path  = next((p for p in [folder / "barcodes.tsv.gz",
                                  folder / "barcodes.tsv"] if p.exists()), None)
    feat_path = next((p for p in [folder / "features.tsv.gz",
                                   folder / "features.tsv",
                                   folder / "genes.tsv.gz",
                                   folder / "genes.tsv"] if p.exists()), None)

    if bc_path is None or feat_path is None:
        raise FileNotFoundError(f"Cannot find barcodes/features in {folder}")

    def _read_tsv(p):
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rt") as f:
            return [line.strip().split("\t") for line in f]

    logger.info("Reading MEX matrix from %s", mtx_path)
    mat = sp.io.mmread(mtx_path).T.tocsr().astype(np.float32)   # cells × genes

    barcodes = [row[0] for row in _read_tsv(bc_path)]
    features = _read_tsv(feat_path)
    gene_names = [r[1] if len(r) > 1 else r[0] for r in features]
    gene_ids   = [r[0] for r in features]

    adata = ad.AnnData(X=mat)
    adata.obs_names = barcodes
    adata.var_names = gene_names
    adata.var["gene_id"] = gene_ids
    logger.info("Loaded MEX: %d cells × %d genes", *mat.shape)
    return adata


def load_h5ad(path: Union[str, Path]) -> ad.AnnData:
    logger.info("Loading h5ad: %s", path)
    adata = ad.read_h5ad(path)
    logger.info("Loaded h5ad: %d cells × %d genes", *adata.shape)
    return adata


def load_csv(path: Union[str, Path], genes_as_rows: bool = False) -> ad.AnnData:
    """
    Load dense CSV/TSV. Assumes cells × genes unless genes_as_rows=True.
    First column treated as index.
    """
    sep = "\t" if str(path).endswith((".tsv", ".tsv.gz")) else ","
    df  = pd.read_csv(path, sep=sep, index_col=0)
    if genes_as_rows:
        df = df.T
    adata = ad.AnnData(X=df.values.astype(np.float32))
    adata.obs_names = list(df.index)
    adata.var_names = list(df.columns)
    logger.info("Loaded CSV: %d cells × %d genes", *adata.shape)
    return adata


def auto_load(path: Union[str, Path]) -> ad.AnnData:
    """
    Detect format and load.
    - Directory  → 10x MEX
    - *.h5ad     → AnnData HDF5
    - *.csv/tsv  → dense matrix
    """
    path = Path(path)
    if path.is_dir():
        return load_10x_mex(path)
    ext = "".join(path.suffixes).lower()
    if ".h5ad" in ext:
        return load_h5ad(path)
    if ".csv" in ext or ".tsv" in ext:
        return load_csv(path)
    raise ValueError(f"Unrecognised format for path: {path}")


# Preprocessing 

def preprocess(
    adata: ad.AnnData,
    min_cells: int = 10,
    min_genes: int = 200,
    max_mito_pct: float = 0.20,
    target_sum: float = 1e4,
    log1p: bool = True,
) -> ad.AnnData:
    """
    Standard scRNA-seq preprocessing:
      1. Filter low-quality cells/genes
      2. Optional mitochondrial QC
      3. Normalize per-cell to target_sum (library-size normalization)
      4. log1p transform
    Returns a new AnnData (original untouched).
    """
    import copy
    adata = copy.copy(adata)
    X = _to_dense_numpy(adata.X)

    # Gene filter
    gene_mask = (X > 0).sum(axis=0) >= min_cells
    X = X[:, gene_mask]
    adata = adata[:, gene_mask].copy()
    logger.info("After gene filter: %d genes retained", X.shape[1])

    # Cell filter
    cell_mask = (X > 0).sum(axis=1) >= min_genes
    X = X[cell_mask, :]
    adata = adata[cell_mask, :].copy()
    logger.info("After cell filter: %d cells retained", X.shape[0])

    # Mitochondrial QC 
    mt_genes = np.array([g.upper().startswith("MT-") for g in adata.var_names])
    if mt_genes.sum() > 0:
        mt_pct = X[:, mt_genes].sum(axis=1) / (X.sum(axis=1) + 1e-9)
        cell_mask2 = mt_pct < max_mito_pct
        X = X[cell_mask2, :]
        adata = adata[cell_mask2, :].copy()
        logger.info("After mito filter (<%.0f%%): %d cells retained",
                    max_mito_pct * 100, X.shape[0])

    # Library-size normalization
    lib_size = X.sum(axis=1, keepdims=True) + 1e-9
    X = X / lib_size * target_sum

    # log1p
    if log1p:
        X = np.log1p(X)

    adata.X = X
    adata.uns["preprocessed"] = True
    adata.uns["target_sum"]   = target_sum
    logger.info("Preprocessing complete: %d cells × %d genes", *X.shape)
    return adata


# col(F) / ker(F) partitioning 

# Canonical codon usage table for E. coli K-12 (fraction per amino acid).
# Rare codons (ker(F) proxies): those with codon usage frequency < 0.10.
# This list can be swapped for human or other organism tables.
_ECOLI_RARE_CODONS = {
    "AGA", "AGG",   # Arg rare
    "ATA",           # Ile rare
    "CTA",           # Leu rare
    "CGA",           # Arg rare
    "GGA",           # Gly moderately rare
    "CCC",           # Pro rare
    "ACA",           # Thr rare
}

# Gene names whose codon composition is known to be ker(F)-enriched
# (de-optimized reporters, known rare-codon genes).
# Populated at runtime via CodonPartitioner.register_ker_genes().
_KNOWN_KER_GENES: set[str] = set()


class CodonPartitioner:
    """
    Partitions gene expression matrix into col(F) and ker(F) sub-matrices.

    col(F): genes whose expression is primarily driven by amino-acid-level
            selection (standard codon-optimized or housekeeping genes).
    ker(F): genes whose synonymous codon usage carries regulatory information
            (rare-codon enriched, wobble-position de-optimized).

    In a physical dual-reporter experiment the partition is explicit
    (mCherry = col(F), GFP-rare = ker(F)). For general scRNA-seq data,
    partition is inferred from codon adaptation index (CAI) or a
    user-supplied gene list.
    """

    def __init__(
        self,
        ker_genes: Optional[list[str]] = None,
        col_genes: Optional[list[str]] = None,
    ):
        self.ker_genes: set[str] = set(ker_genes or []) | _KNOWN_KER_GENES
        self.col_genes: set[str] = set(col_genes or [])

    def register_ker_genes(self, genes: list[str]) -> "CodonPartitioner":
        self.ker_genes.update(genes)
        return self

    def register_col_genes(self, genes: list[str]) -> "CodonPartitioner":
        self.col_genes.update(genes)
        return self

    def partition(
        self,
        adata: ad.AnnData,
        device: Optional[torch.device] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with keys:
          'X_full'   — full expression tensor  (cells × genes)
          'X_col'    — col(F) sub-matrix
          'X_ker'    — ker(F) sub-matrix
          'col_idx'  — gene indices for col
          'ker_idx'  — gene indices for ker
          'gene_names' — full list
        """
        if device is None:
            device = _get_device()

        X = _to_dense_numpy(adata.X)
        genes = list(adata.var_names)

        col_idx = np.array([i for i, g in enumerate(genes)
                            if g in self.col_genes], dtype=np.int64)
        ker_idx = np.array([i for i, g in enumerate(genes)
                            if g in self.ker_genes], dtype=np.int64)

        # Fallback: if no explicit labels, use expression variance as a proxy.
        # High-variance genes → col(F) (strongly selected).
        # Low-variance but expressed genes → ker(F) (wobble-buffered).
        if len(col_idx) == 0 and len(ker_idx) == 0:
            logger.warning(
                "No col/ker genes registered. Using variance-based proxy split."
            )
            var = X.var(axis=0)
            median_var = np.median(var[var > 0])
            col_idx = np.where(var >= median_var)[0]
            ker_idx = np.where((var > 0) & (var < median_var))[0]
            logger.info(
                "Variance proxy: %d col(F) genes, %d ker(F) genes",
                len(col_idx), len(ker_idx),
            )

        X_t   = torch.tensor(X,              dtype=torch.float32, device=device)
        X_col = X_t[:, col_idx] if len(col_idx) > 0 else torch.zeros(X_t.shape[0], 0, device=device)
        X_ker = X_t[:, ker_idx] if len(ker_idx) > 0 else torch.zeros(X_t.shape[0], 0, device=device)

        logger.info(
            "Partitioned: %d col(F) genes, %d ker(F) genes (device=%s)",
            len(col_idx), len(ker_idx), device,
        )
        return {
            "X_full":     X_t,
            "X_col":      X_col,
            "X_ker":      X_ker,
            "col_idx":    col_idx,
            "ker_idx":    ker_idx,
            "gene_names": genes,
        }
