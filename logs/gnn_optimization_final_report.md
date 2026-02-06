# GNN Optimization - Final Results Report

**Date:** 2026-02-04
**Experiment:** Optimized GCN with increased genes and relaxed STRING threshold

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Mean AUROC** | **0.6851 ± 0.0914** |
| **Best Fold AUROC** | **0.8107** (Fold 2) |
| Mean Accuracy | 0.6145 ± 0.0776 |

### Key Finding
The optimized GCN shows **high variance** across folds (0.58 - 0.81). Fold 2 achieved excellent performance (0.81), approaching the LR baseline (0.82), while Fold 4 underperformed (0.58). This suggests the model is sensitive to the training data split.

---

## Configuration

### Graph Construction
| Parameter | Previous | Optimized |
|-----------|----------|-----------|
| Variance genes | 500 | **2000** |
| STRING threshold | 0.9 | **0.7** |
| Final nodes | 1,050 | **1,491** |
| Edges | ~10,700 | **18,482** |
| Avg degree | 28.3 | **24.8** |

### Model Architecture
| Parameter | Previous | Optimized |
|-----------|----------|-----------|
| Hidden Channels | 32 | **64** |
| Layers | 2 | **3** |
| Dropout | 0.7 | **0.5** |
| Edge Dropout | 5% | **10%** |
| Feature Noise | 0 | **0.1** |
| Epochs | 100 | **150** |
| Learning Rate | 0.0005 | **0.001** |

---

## Cross-Validation Results

| Fold | AUROC | Accuracy | Notes |
|------|-------|----------|-------|
| 1 | 0.6435 | 62.3% | |
| 2 | **0.8107** | 68.1% | ★ Best fold |
| 3 | 0.6167 | 59.4% | |
| 4 | 0.5786 | 47.8% | Worst fold |
| 5 | 0.7761 | 69.6% | |

---

## Model Comparison

| Model | Mean AUROC | Std Dev | Best Fold | Status |
|-------|------------|---------|-----------|--------|
| **LR Baseline (5000 genes)** | **0.8164** | 0.0744 | - | ✅ Best overall |
| RF Baseline (10000 genes) | 0.7927 | 0.0657 | - | |
| GCN Optimized | 0.6851 | 0.0914 | 0.8107 | High variance |
| Previous GCN | 0.6812 | 0.0478 | 0.7422 | More stable |

### Interpretation
1. **LR baseline remains superior** for mean performance (0.82 vs 0.69)
2. **GCN shows potential** - Fold 2 achieved 0.81, proving the architecture can work
3. **High variance is the main issue** - The model is overfitting to specific data splits
4. **Previous GCN was more stable** (std 0.05 vs 0.09)

---

## Saved Artifacts

### Models (`gnn_optimized/models/`)
- `gcn_best.pt` - Best overall model (AUROC=0.8107, Fold 2)
- `gcn_fold1.pt` through `gcn_fold5.pt` - Per-fold best models

### Data (`gnn_optimized/data/`)
- `patient_graphs_optimized.pkl` - 345 patient graphs with 1,491 nodes

---

## Recommendations for Improvement

1. **Reduce model complexity** - 3 layers may be too deep; try 2 layers
2. **Increase regularization** - Higher dropout (0.6-0.7) to reduce variance
3. **Try GraphSAGE** - Different architecture may be more robust
4. **Ensemble models** - Average predictions from multiple folds
5. **Feature engineering** - Add more node features (pathway membership, etc.)

---

## Next Steps

- [ ] Train GraphSAGE with same configuration
- [ ] External validation on GSE26440 (Pediatric)
- [ ] GNNExplainer analysis on best model
