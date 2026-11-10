# Experiment `checks_b`

One shared normalization per workflow across all policies below. IGD ↓ · HV_jin ↑ · Imprecision ↓ · PIBI ↓ · HV_hi ↑. Agent policies: each eval seed is averaged over training seeds. Symbols: Wilcoxon signed-rank test (paired by seed, 5%) against *ip_moea_hv* — `+` better, `−` worse, `=` no significant difference.

## Overall (mean over workflows; rank by IGD)

| Policy | mean rank | IGD | HV_jin | Imprecision | PIBI | HV_hi | +/−/= vs ip_moea_hv (IGD, HV_jin, Imprecision) |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 1.00 | 0.02254 | 0.7284 | 111.8 | 1.809 | 0.3589 | (reference) |
| imoea_dqn | 2.00 | 0.2237 | 0.6939 | 43.18 | 2.119 | 0.2556 | 0/3/0 · 0/2/1 · 3/0/0 |

## montage_400@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.0154 ± 0.0036 | 0.7164 ± 0.02 | 115.4 ± 3.8 | 2.089 ± 0.012 | 0.2942 ± 0.01 | 104 | 1.00 |
| imoea_dqn | 0.1385 ± 0.099 − | 0.7031 ± 0.021 − | 53.02 ± 22 + | 2.159 ± 0.077 − | 0.2485 ± 0.023 − | 59 | 2.00 |

## cybershake_1000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.03423 ± 0.0036 | 0.7292 ± 0.033 | 102.4 ± 3.8 | 1.664 ± 0.073 | 0.4104 ± 0.03 | 104 | 1.00 |
| imoea_dqn | 0.2613 ± 0.1 − | 0.6988 ± 0.056 = | 39.8 ± 18 + | 2.068 ± 0.2 − | 0.2834 ± 0.071 − | 42 | 2.00 |

## epigenomics_1000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.018 ± 0.0049 | 0.7395 ± 0.05 | 117.8 ± 4.5 | 1.676 ± 0.11 | 0.372 ± 0.041 | 104 | 1.00 |
| imoea_dqn | 0.2714 ± 0.099 − | 0.68 ± 0.049 − | 36.72 ± 17 + | 2.129 ± 0.16 − | 0.2348 ± 0.059 − | 36 | 2.00 |
