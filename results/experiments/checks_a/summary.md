# Experiment `checks_a`

One shared normalization per workflow across all policies below. IGD ↓ · HV_jin ↑ · Imprecision ↓ · PIBI ↓ · HV_hi ↑. Agent policies: each eval seed is averaged over training seeds. Symbols: Wilcoxon signed-rank test (paired by seed, 5%) against *ip_moea_hv* — `+` better, `−` worse, `=` no significant difference.

## Overall (mean over workflows; rank by IGD)

| Policy | mean rank | IGD | HV_jin | Imprecision | PIBI | HV_hi | +/−/= vs ip_moea_hv (IGD, HV_jin, Imprecision) |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 1.00 | 0.05367 | 0.6099 | 72.36 | 2.119 | 0.3821 | (reference) |
| imoea_dqn | 2.00 | 0.2592 | 0.5954 | 41.24 | 2.433 | 0.3264 | 0/1/0 · 0/0/1 · 1/0/0 |

## medium_250_i0@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.05367 ± 0.01 | 0.6099 ± 0.023 | 72.36 ± 0.98 | 2.119 ± 0.051 | 0.3821 ± 0.017 | 105 | 1.00 |
| imoea_dqn | 0.2592 ± 0.13 − | 0.5954 ± 0.064 = | 41.24 ± 13 + | 2.433 ± 0.22 − | 0.3264 ± 0.058 − | 62 | 2.00 |
