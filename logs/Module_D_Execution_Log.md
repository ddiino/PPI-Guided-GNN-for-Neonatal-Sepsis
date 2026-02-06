# Module D Execution Log

## Final Results (2026-01-27) ✅

### Baseline Benchmark (Phase 1.5 - Corrected Data)
| Model | AUROC | Accuracy |
|-------|-------|----------|
| Random Forest | 0.5477 | 55.8% |
| Logistic Regression | 0.5758 | 57.1% |

**Number to beat: 0.5758 (Logistic Regression)**

---

### GCN Training (with weight saving)
- Mean AUROC: **0.6832** (+/- 0.0509) 
- Best Fold: **0.7713**
- Folds: [0.694, 0.679, 0.771, 0.619, 0.653]
- **Verdict:** ✓ GCN BEATS BASELINE (+12.0%)

### GAT Training (with weight saving)
- Mean AUROC: **0.6350** (+/- 0.0214)
- Best Fold: **0.6647**
- Folds: [0.639, 0.665, -, 0.605, 0.618]
- **Verdict:** ✓ GAT BEATS BASELINE (+7.2%)

---

## Saved Model Weights
```
models/
├── gcn_best.pt      (12.7 KB, AUROC=0.771)
├── gcn_fold1-5.pt   (5 files)
├── gat_best.pt      (24.3 KB, AUROC=0.665)
└── gat_fold1-5.pt   (5 files)
```

---

## Summary

| Model | AUROC | vs Baseline |
|-------|-------|-------------|
| **GCN** | 0.6832 | **+12.0%** |
| GAT | 0.6350 | +7.2% |

**Module D Status:** 🔄 IN PROGRESS (Retraining with Fixes)

---

## Phase 1.5: Strategic Model Refinement (2026-01-28)

### Hyperparameter Adjustments
- **Learning Rate:** Reduced from `0.001` to **0.0005** to improve stability with multi-platform data.
- **Training Set:** Strictly defined as GSE25504 (Affy/Illu) + GSE69686 (319 samples).
- **Validation:** GSE26440 kept strictly external.

### Results (2026-01-28)
- **GCN (Corrected Data):**
  - Mean AUROC: **0.6812** (+/- 0.0478)
  - Best Fold: **0.7422**
  - **Status:** BEST PERFORMER (Selected for Validation)
- **GAT (Corrected Data):**
  - Mean AUROC: **0.6345** (+/- 0.0689)
  - Best Fold: **0.7448** (High variance, unstable)

### Action
- **Weights:** Saved to `models/gcn_best_merged.pt` and `models/gat_best_merged.pt`.
- **Verdict:** Proceeding to External Validation with GCN.
