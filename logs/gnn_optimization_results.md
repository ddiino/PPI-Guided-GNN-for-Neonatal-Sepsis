# GCN Optimized Results

**Date:** 2026-02-04 21:47

## Configuration

| Parameter | Value |
|-----------|-------|
| Hidden Channels | 64 |
| Num Layers | 3 |
| Dropout | 0.5 |
| Edge Dropout | 0.1 |
| Feature Noise | 0.1 |
| Learning Rate | 0.001 |
| Weight Decay | 0.0005 |
| Batch Size | 32 |
| Epochs | 150 |
| Nodes | 1491 |
| Edges | 18482 |

## Results

| Fold | AUROC | Accuracy |
|------|-------|----------|
| 1 | 0.6435 | 0.6232 |
| 2 | 0.8107 | 0.6812 |
| 3 | 0.6167 | 0.5942 |
| 4 | 0.5786 | 0.4783 |
| 5 | 0.7761 | 0.6957 |

## Summary

| Metric | Value |
|--------|-------|
| **Mean AUROC** | **0.6851 ± 0.0914** |
| Mean Accuracy | 0.6145 ± 0.0776 |
| Best Fold | 0.8107 (Fold 2) |

## Comparison with Baselines

| Model | AUROC |
|-------|-------|
| GCN Optimized | 0.6851 |
| Previous GCN | 0.6812 |
| LR Baseline | 0.8164 |
