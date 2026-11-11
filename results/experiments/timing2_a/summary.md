# Experiment `timing2_a`

One shared normalization per workflow across all policies below. IGD ↓ · HV_jin ↑ · Imprecision ↓ · PIBI ↓ · HV_hi ↑. Agent policies: each eval seed is averaged over training seeds. Symbols: Wilcoxon signed-rank test (paired by seed, 5%) against *final* — `+` better, `−` worse, `=` no significant difference.

## Overall (mean over workflows; rank by IGD)

| Policy | mean rank | IGD | HV_jin | Imprecision | PIBI | HV_hi | +/−/= vs final (IGD, HV_jin, Imprecision) |
|---|---|---|---|---|---|---|---|
| final | 1.00 | 0.008346 | 0.8102 | 33.48 | 1.942 | 0.6842 | (reference) |
| ip_moea_hv | 2.00 | 0.1178 | 0.6597 | 29.59 | 2.065 | 0.5603 | 0/0/2 · 0/0/2 · 0/0/2 |
| imoea_dqn | 3.00 | 0.2447 | 0.5725 | 15.88 | 2.512 | 0.4597 | 0/0/2 · 0/0/2 · 0/0/2 |

## small_20_i0@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| final | 0.009263 ± 0.00054 | 0.822 ± 0.001 | 26.62 ± 0.76 | 1.929 ± 0.0015 | 0.724 ± 0.0012 | 96 | 1.00 |
| ip_moea_hv | 0.0955 ± 0.019 = | 0.7121 ± 0.018 = | 23.62 ± 0.53 = | 2.001 ± 0.026 = | 0.6345 ± 0.017 = | 104 | 2.00 |
| imoea_dqn | 0.2688 ± 0.16 = | 0.5882 ± 0.12 = | 16.3 ± 2.4 = | 2.525 ± 0.43 = | 0.4984 ± 0.12 = | 84 | 3.00 |

## medium_250_i0@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| final | 0.00743 ± 0.00063 | 0.7984 ± 0.0013 | 40.35 ± 0.56 | 1.956 ± 0.0016 | 0.6444 ± 0.00077 | 101 | 1.00 |
| ip_moea_hv | 0.1402 ± 0.0085 = | 0.6073 ± 0.007 = | 35.56 ± 0.28 = | 2.129 ± 0.024 = | 0.486 ± 0.0072 = | 105 | 2.00 |
| imoea_dqn | 0.2207 ± 0.056 = | 0.5567 ± 0.045 = | 15.45 ± 6 = | 2.5 ± 0.17 = | 0.421 ± 0.045 = | 44 | 3.00 |
