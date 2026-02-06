# Module D Results Summary

## Final Model Comparison

| Model | Mean AUROC | Std Dev | Best Fold | vs Baseline |
|-------|------------|---------|-----------|-------------|
| **GCN** | **0.6832** | ±0.0509 | 0.771 | **+12.0%** ✓ |
| GAT | 0.6350 | ±0.0214 | 0.665 | +7.2% ✓ |
| Random Forest | 0.5628 | ±0.0759 | - | Baseline |
| Logistic Regression | 0.4786 | ±0.0266 | - | - |

## Key Finding
**GRAPH STRUCTURE PROVIDES LIFT.** Both GNN models outperform baselines.

## Saved Model Weights

| File | Description | AUROC |
|------|-------------|-------|
| `models/gcn_best.pt` | Best GCN (overall) | 0.771 |
| `models/gcn_fold1-5.pt` | Per-fold GCN models | - |
| `models/gat_best.pt` | Best GAT (overall) | 0.665 |
| `models/gat_fold1-5.pt` | Per-fold GAT models | - |

## Cross-Validation Details

**GCN (5-Fold):** [0.694, 0.679, 0.771, 0.619, 0.653]  
**GAT (5-Fold):** [0.639, 0.665, -, 0.605, 0.618]

## Configuration

- **Samples:** 319 (Control: 186, Sepsis: 133)
- **Nodes:** 1,047 (variance-filtered)
- **Features:** 3D (expression, degree, variance)

**GCN:** 2 layers, 64 hidden, 0.5 dropout, 2,402 params  
**GAT:** 2 layers, 64 hidden, 2 heads, 0.6 dropout, 5,026 params

---
*Generated: 2026-01-27*
