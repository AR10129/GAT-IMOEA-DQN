# Experiment `rerun_ipmoea_a`

One shared normalization per workflow across all policies below. IGD ↓ · HV_jin ↑ · Imprecision ↓ · PIBI ↓ · HV_hi ↑. Agent policies: each eval seed is averaged over training seeds. Symbols: Wilcoxon signed-rank test (paired by seed, 5%) against *ip_moea_hv* — `+` better, `−` worse, `=` no significant difference.

## Overall (mean over workflows; rank by IGD)

| Policy | mean rank | IGD | HV_jin | Imprecision | PIBI | HV_hi | +/−/= vs ip_moea_hv (IGD, HV_jin, Imprecision) |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 1.00 | 0.01692 | 0.6592 | 67 | 2.257 | 0.4151 | (reference) |

## small_20_i0@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01584 ± 0.0073 | 0.6992 ± 0.019 | 40.56 ± 1.5 | 2.18 ± 0.0087 | 0.5477 ± 0.014 | 105 | 1.00 |

## small_20_i0@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01582 ± 0.0048 | 0.7248 ± 0.0095 | 45.71 ± 1.2 | 2.1 ± 0.011 | 0.5512 ± 0.0062 | 105 | 1.00 |

## small_50_i0@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.008727 ± 0.0028 | 0.6784 ± 0.014 | 42.02 ± 0.82 | 2.248 ± 0.01 | 0.5164 ± 0.0079 | 104 | 1.00 |

## small_50_i0@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01941 ± 0.0062 | 0.7351 ± 0.022 | 47.43 ± 1.3 | 1.985 ± 0.013 | 0.5653 ± 0.018 | 105 | 1.00 |

## basic_60_i0@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.02014 ± 0.0087 | 0.637 ± 0.04 | 58.59 ± 2.3 | 2.289 ± 0.03 | 0.4212 ± 0.025 | 105 | 1.00 |

## basic_60_i0@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.02462 ± 0.0055 | 0.6596 ± 0.018 | 69.49 ± 1.8 | 2.245 ± 0.014 | 0.3957 ± 0.011 | 105 | 1.00 |

## basic_100_i0@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.0102 ± 0.0048 | 0.6662 ± 0.031 | 69.13 ± 1.2 | 2.294 ± 0.017 | 0.4003 ± 0.017 | 105 | 1.00 |

## basic_100_i0@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.016 ± 0.0035 | 0.6305 ± 0.016 | 83.09 ± 1.1 | 2.331 ± 0.01 | 0.3345 ± 0.0078 | 105 | 1.00 |

## medium_100_i0@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01089 ± 0.0033 | 0.6215 ± 0.034 | 73.77 ± 1.1 | 2.349 ± 0.02 | 0.3523 ± 0.017 | 105 | 1.00 |

## medium_100_i0@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.02851 ± 0.01 | 0.6059 ± 0.037 | 92.18 ± 2.1 | 2.355 ± 0.038 | 0.2902 ± 0.018 | 105 | 1.00 |

## medium_250_i0@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01357 ± 0.0064 | 0.6106 ± 0.022 | 79.36 ± 1.8 | 2.387 ± 0.013 | 0.3246 ± 0.0093 | 105 | 1.00 |

## medium_250_i0@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01928 ± 0.0048 | 0.6412 ± 0.022 | 102.7 ± 1.3 | 2.324 ± 0.022 | 0.2817 ± 0.01 | 105 | 1.00 |
