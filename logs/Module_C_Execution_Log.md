# Module C Execution Log

## Task C.1: Network Construction & Graph Generation

### Phase 1: Implementation
- **PPI Network:** Parsed STRING v12 interaction data.
- **Filtering:** Filtered interactions with confidence score > 400.
- **Node Features:** Mapped patient gene expression data to network nodes.
- **Graph Generation:** Constructed separate graph objects for each patient sample.

### Phase 2: Verification (CoVe)
- **Status:** PASSED
- **Output File:** `data/graphs/patient_graphs.pkl`
- **File Size:** 16.6 MB
- **Graph Structure:**
  - Nodes: 9,380 genes
  - Edges: 265,712 (bidirectional)
  - Attributes: Z-score normalized expression values

## Task C.2: Baseline Modeling

### Phase 1: Implementation
- Implemented standard machine learning baselines for comparison.
- **Models:** Random Forest, Logistic Regression.
- **Method:** Trained on tabular expression data (flattened features).

### Phase 2: Verification (CoVe)
- **Status:** PASSED
- **Results (`data/processed/baseline_results.csv`):**
  - **Random Forest:** AUROC = 0.808, Accuracy = 72.6%
  - **Logistic Regression:** AUROC = 0.856, Accuracy = 73.6%

---

## Enhancement (2026-01-27): 3D Node Features + Variance Filtering

### Feature Selection (CRITICAL FIX)
- **Problem:** 9,380 nodes with 319 samples = 1:30 ratio (curse of dimensionality)
- **Solution:** Select top 2,000 genes by variance, intersect with PPI network
- **Result:** 1,047 nodes (only high-variance genes with network edges)

### 3D Node Features
| Feature | Source | Normalization |
|---------|--------|---------------|
| Expression | Per-sample | Z-score |
| Degree Centrality | PPI graph | Global Z-score |
| Gene Variance | Across cohort | Global Z-score |

### Verification (CoVe)
- **Output:** `data/graphs/patient_graphs_3d.pkl`
- **Shape:** 319 graphs × **1,047 nodes** × 3 features ✓
- **Edges:** 10,700 (bidirectional)
- **Sample:Feature Ratio:** 1:3.3 (was 1:30) ✓
- **Labels:** Control=186, Sepsis=133

---

**Module C Status:** ✅ COMPLETE (Variance-filtered 2026-01-27)

---

## Phase 1.5: Strategic Graph Optimization (2026-01-28)

### Adaptive Thresholding
- **Goal:** Ensure graph connectivity despite platform variation.
- **Method:** Iterative threshold selection (0.9 → 0.4) targeting Avg Degree ≥ 5.
- **Outcome:** Selected threshold **0.9** (Avg Degree = 28.3).
- **Global Graph:** 9,380 nodes, 66,404 edges (Coverage: 60%).

### Patient Graph Regeneration
- **Script:** `04_create_graphs_variance_filtered.py` using new ComBat data.
- **Output:** 319 patient graphs (`patient_graphs_3d.pkl`).
- **Features:** 1,050 nodes (Variance-filtered ∩ Network) with 3D features (Expr, Degree, Variance).

