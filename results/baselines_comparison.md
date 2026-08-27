# Phase 8 — Baseline Comparison

## Concept Vector Quality: Val |Pearson| (best block)

| Method | food | valence | env_social |
|--------|------|---------|------------|
| pca        | 0.9288 | 0.1544 | 0.4900 |
| diffmeans  | 0.9749 | 0.7995 | 0.9142 |
| logreg     | 0.9857 | 0.9603 | 0.9721 |


## Monitoring AUROC Comparison

*(Aggregate probe; test split)*

| Method | food | valence | env_social | Overall |
|--------|------|---------|------------|---------|
| PCA        | 0.9947 | 0.7425 | 0.8992 | 0.8788 |
| DiffMeans  | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LogReg     | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RFM        | 1.0000 | 1.0000 | 1.0000 | 1.0000 |


## Steering Success-Rate Comparison

*(Best-over-positive-ε; probe-score > 0.5 OR lexicon hit)*

| Method | food | valence | env_social | Overall |
|--------|------|---------|------------|---------|
| PCA        | 1.000 | 1.000 | 1.000 | 1.000 |
| DiffMeans  | 1.000 | 1.000 | 1.000 | 1.000 |
| LogReg     | 1.000 | 1.000 | 1.000 | 1.000 |
| RFM        | 1.000 | 1.000 | 1.000 | 1.000 |


### Winner summary

- Best monitoring AUROC: **DiffMeans**
- Best steering success: **PCA**

*Note: RFM monitoring uses 3 eigenvectors per block (Phase 7); baselines use 1 vector per block.*  
*RFM steering uses vectors from 01_extract.py; success rates come from eval6 Phase 6 results.*  
*All methods evaluated on identical splits, identical ε sweeps, identical probe threshold (0.5).*
