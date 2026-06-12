# Quantum Bio-Seam Verification Pipeline

PyTorch verification suite for **The Discrete col(F)/ker(F) Boundary as the Quantum-Classical Interface of Biological Information**.

ERI Labs · Eric Ren · June 2026

---

## What this proves

| Proof | Claim | Expected result |
|---|---|---|
| 1 — Sherman-Morrison Efficiency | O(N²) rank-1 updates vs O(N³) re-inversion | ~100× speedup on GPU |
| 2 — Architecture Gap | ker(F) signal invisible to AA-supervised MLP | ~65× MSE gap |
| 3 — SVD Rank-1 | ker(F) tRNA bottleneck propagates as rank-1 update | First mode >90% of ΔA variance |

---

## Install

```bash
pip install torch numpy pandas scipy matplotlib seaborn anndata
```

---

## Run

```bash
# Synthetic data (no files needed)
python run_pipeline.py

# Real scRNA-seq
python run_pipeline.py --scrna your_data.h5ad \
    --ker-genes GFP_deopt,rare_codon_reporter \
    --col-genes mCherry,ACTB,GAPDH

# Scale up
python run_pipeline.py --n-genes 20000 --grn-genes 5000 --epochs 800
```

Outputs go to `outputs/plots/` and `outputs/results/`.

---

## Structure

```
src/
  ingest.py            — scRNA-seq ingestion, QC, col/ker partitioning
  sherman_morrison.py  — GRN engine, rank-1 updates, SVD validator
  models.py            — Kernel-aware MLP vs continuous MLP
  viz.py               — All figures
run_pipeline.py        — Master runner
```

---

## Upstream theory

[LAB-GUIDE-THE-QUANTUM-BIO-SEAM](https://github.com/ericrenone/LAB-GUIDE-THE-QUANTUM-BIO-SEAM)
