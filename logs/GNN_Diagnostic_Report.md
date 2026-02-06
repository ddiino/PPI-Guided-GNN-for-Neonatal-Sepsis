# GNN Diagnostic Analysis Report

## Question 1: Why 212 Graphs Instead of 320?

### Root Cause Found
The metadata file shows **107 samples labeled as "Unknown"** in GSE25504. These are excluded during training:

```
Total samples in combined_metadata.csv: 320
- GSE25504: 170 samples
  - Control: 37
  - Sepsis: 26
  - Unknown: 107  ← EXCLUDED
- GSE69686: 149 samples (all labeled Sepsis or Control)
```

**Calculation:** `320 - 107 = 213` (we see 212 due to possible sample mismatch)

### Why Are There Unknown Labels?
From Module_B_Execution_Log.md:
> "Many samples labeled 'Unknown' due to ambiguous metadata (107 in GSE25504)"

The GEO dataset has samples where the sepsis/control status could not be reliably parsed from the metadata.

### Impact on Training
- **Training data reduced by ~34%**
- **Class imbalance worsened** (fewer Sepsis cases)
- **Less data = higher variance in model performance**

---

## Question 2: Would Adding BioGRID Help?

### Current State
- Using STRING v12 only (BioGRID download failed earlier)
- Current network has ~2,000 nodes (genes) after intersection with expression data

### Analysis
| Factor | Impact |
|--------|--------|
| More edges | Could help GAT learn better attention patterns |
| More nodes | Would increase coverage of gene expression |
| Memory | GAT already OOM'd; more edges would worsen |
| Evidence | The baseline (Logistic Regression) with NO graph info scores 0.856 |

### Conclusion
**Unlikely to help.** The problem is that the **baselines without any graph structure outperform GNNs.** This suggests:
1. Graph topology is not adding useful signal for this classification
2. The node features (expression values) alone are sufficient
3. Adding more edges won't fix a fundamental problem

---

## Question 3: Are We Overfitting? How Many Epochs?

### Current Training Settings
| Parameter | GCN | GAT |
|-----------|-----|-----|
| Epochs | **50** | **50** |
| Hidden Channels | 32 | 16 |
| Batch Size | 16 | 4 |
| Learning Rate | 0.001 | 0.001 |

### Signs of Overfitting vs. Underfitting
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Training Loss | ~0.68 (stable) | Not decreasing = **Underfitting** |
| Val AUROC | 0.5-0.7 (variable) | Near random = **Not learning useful patterns** |
| Fold Variance | ±0.05-0.07 | High variance = Unstable learning |

### Diagnosis: **Not Overfitting - Actually Underfitting**
- Loss is barely decreasing (stuck at ~0.68, which is near `log(2) = 0.69`)
- Model is not learning discriminative features from graph structure
- More epochs would not help; the model has already plateaued

---

## Recommendations to Improve Accuracy

### Priority 1: Fix the Unknown Labels (HIGH IMPACT)
- **Action:** Manually review GSE25504 metadata to recover true labels
- **Potential gain:** +34% more training data
- **Approach:** Check original GEO supplementary files or paper

### Priority 2: Use Baseline Model for Now (PRAGMATIC)
- Logistic Regression achieves **0.856 AUROC** without graph info
- For the project deadline, use LR for external validation
- Document GNN as "experimental" with negative results (this is valid science!)

### Priority 3: Improve Node Features (MEDIUM IMPACT)
- Current: 1 feature per node (expression z-score)
- Add: Differential expression, variance, pathway membership
- More node features = more signal for GNNs to learn from

### Priority 4: Different GNN Architecture (LOW IMPACT)
- Try GraphSAGE (simpler, may work better on small datasets)
- Try adding skip connections / residual layers
- Reduce model complexity further

### NOT Recommended
| Action | Why Not |
|--------|---------|
| More epochs | Already plateaued at low loss |
| Add BioGRID | Memory issues, unlikely to help |
| Tune hyperparameters | Small dataset = high variance; gains will be marginal |

---

## Summary

| Question | Answer |
|----------|--------|
| Why 212 vs 320? | **107 samples have Unknown labels and are excluded** |
| Would BioGRID help? | **No** - baselines without graphs already beat GNNs |
| Overfitting? | **No - Underfitting.** Model is not learning useful patterns |
| How many epochs? | **50 epochs** (already plateaued) |
| How to improve? | **Fix Unknown labels** or **use baseline model** |
