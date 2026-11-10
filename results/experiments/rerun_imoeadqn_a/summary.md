# Experiment `rerun_imoeadqn_a`

One shared normalization per workflow across all policies below. IGD ↓ · HV_jin ↑ · Imprecision ↓ · PIBI ↓ · HV_hi ↑. Agent policies: each eval seed is averaged over training seeds. Symbols: Wilcoxon signed-rank test (paired by seed, 5%) against *imoea_dqn* — `+` better, `−` worse, `=` no significant difference.

## Overall (mean over workflows; rank by IGD)

| Policy | mean rank | IGD | HV_jin | Imprecision | PIBI | HV_hi | +/−/= vs imoea_dqn (IGD, HV_jin, Imprecision) |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 1.00 | 0.09334 | 0.7006 | 29.19 | 2.183 | 0.4713 | (reference) |

## small_20_i0@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.05185 ± 0.025 | 0.7811 ± 0.026 | 15.13 ± 5.9 | 1.978 ± 0.038 | 0.6626 ± 0.023 | 54 | 1.00 |

## small_20_i0@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.169 ± 0.15 | 0.6578 ± 0.095 | 20.26 ± 5.6 | 2.293 ± 0.32 | 0.5277 ± 0.092 | 72 | 1.00 |

## small_50_i0@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.09987 ± 0.078 | 0.71 ± 0.082 | 19.78 ± 5.1 | 2.149 ± 0.2 | 0.5816 ± 0.078 | 75 | 1.00 |

## small_50_i0@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.08436 ± 0.056 | 0.6968 ± 0.058 | 25.49 ± 6.5 | 2.124 ± 0.16 | 0.5517 ± 0.056 | 73 | 1.00 |

## basic_60_i0@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.07545 ± 0.041 | 0.7017 ± 0.026 | 24.52 ± 8.4 | 2.22 ± 0.042 | 0.4641 ± 0.025 | 52 | 1.00 |

## basic_60_i0@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.06042 ± 0.035 | 0.725 ± 0.035 | 38.38 ± 12 | 2.13 ± 0.11 | 0.4461 ± 0.037 | 64 | 1.00 |

## basic_100_i0@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.09093 ± 0.099 | 0.7153 ± 0.07 | 31.05 ± 11 | 2.188 ± 0.21 | 0.4623 ± 0.071 | 63 | 1.00 |

## basic_100_i0@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.1372 ± 0.11 | 0.5903 ± 0.099 | 33.06 ± 7.4 | 2.489 ± 0.32 | 0.318 ± 0.091 | 54 | 1.00 |

## medium_100_i0@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.08193 ± 0.071 | 0.7135 ± 0.034 | 31.45 ± 9.5 | 2.212 ± 0.053 | 0.4079 ± 0.032 | 51 | 1.00 |

## medium_100_i0@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.09724 ± 0.072 | 0.7027 ± 0.077 | 34.45 ± 13 | 2.145 ± 0.2 | 0.412 ± 0.071 | 56 | 1.00 |

## medium_250_i0@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.07581 ± 0.039 | 0.7562 ± 0.05 | 31.87 ± 11 | 2.016 ± 0.12 | 0.4696 ± 0.048 | 59 | 1.00 |

## medium_250_i0@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.096 ± 0.075 | 0.6564 ± 0.07 | 44.78 ± 14 | 2.254 ± 0.22 | 0.352 ± 0.063 | 62 | 1.00 |
