# Experiment `rerun_imoeadqn_b`

One shared normalization per workflow across all policies below. IGD ↓ · HV_jin ↑ · Imprecision ↓ · PIBI ↓ · HV_hi ↑. Agent policies: each eval seed is averaged over training seeds. Symbols: Wilcoxon signed-rank test (paired by seed, 5%) against *imoea_dqn* — `+` better, `−` worse, `=` no significant difference.

## Overall (mean over workflows; rank by IGD)

| Policy | mean rank | IGD | HV_jin | Imprecision | PIBI | HV_hi | +/−/= vs imoea_dqn (IGD, HV_jin, Imprecision) |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 1.00 | 0.08496 | 0.7781 | 47.64 | 1.904 | 0.3655 | (reference) |

## cybershake_100@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.1302 ± 0.14 | 0.7524 ± 0.072 | 48.75 ± 20 | 2.062 ± 0.18 | 0.4379 ± 0.072 | 75 | 1.00 |

## cybershake_100@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.08603 ± 0.063 | 0.805 ± 0.058 | 36.32 ± 9.8 | 1.866 ± 0.17 | 0.5394 ± 0.058 | 71 | 1.00 |

## cybershake_100@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.07731 ± 0.048 | 0.8088 ± 0.068 | 50.29 ± 9.4 | 1.725 ± 0.2 | 0.512 ± 0.069 | 77 | 1.00 |

## cybershake_400@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.0714 ± 0.043 | 0.777 ± 0.047 | 37.11 ± 19 | 2.003 ± 0.14 | 0.3544 ± 0.062 | 47 | 1.00 |

## cybershake_400@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.06534 ± 0.021 | 0.7782 ± 0.024 | 50.56 ± 19 | 1.961 ± 0.079 | 0.3295 ± 0.033 | 52 | 1.00 |

## cybershake_400@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.1751 ± 0.14 | 0.7634 ± 0.042 | 70.77 ± 20 | 1.877 ± 0.12 | 0.3586 ± 0.044 | 74 | 1.00 |

## cybershake_1000@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.09632 ± 0.045 | 0.7892 ± 0.045 | 31.21 ± 15 | 1.915 ± 0.14 | 0.3258 ± 0.056 | 32 | 1.00 |

## cybershake_1000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.1174 ± 0.09 | 0.7688 ± 0.062 | 44.92 ± 20 | 1.935 ± 0.17 | 0.2984 ± 0.078 | 42 | 1.00 |

## cybershake_1000@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.1223 ± 0.071 | 0.8107 ± 0.067 | 48.98 ± 20 | 1.742 ± 0.21 | 0.3651 ± 0.089 | 48 | 1.00 |

## epigenomics_100@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.08192 ± 0.063 | 0.7585 ± 0.065 | 35.05 ± 9.6 | 2.046 ± 0.16 | 0.4345 ± 0.068 | 58 | 1.00 |

## epigenomics_100@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.07188 ± 0.05 | 0.8068 ± 0.053 | 44.69 ± 11 | 1.804 ± 0.13 | 0.5369 ± 0.055 | 83 | 1.00 |

## epigenomics_100@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.08825 ± 0.088 | 0.8344 ± 0.056 | 48.77 ± 8.9 | 1.601 ± 0.18 | 0.5543 ± 0.058 | 64 | 1.00 |

## epigenomics_400@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.04284 ± 0.041 | 0.8386 ± 0.029 | 40.63 ± 21 | 1.823 ± 0.065 | 0.41 ± 0.035 | 50 | 1.00 |

## epigenomics_400@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.08643 ± 0.084 | 0.7738 ± 0.022 | 63.87 ± 23 | 1.933 ± 0.049 | 0.3092 ± 0.022 | 64 | 1.00 |

## epigenomics_400@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.0758 ± 0.033 | 0.737 ± 0.023 | 63.23 ± 33 | 1.968 ± 0.077 | 0.2376 ± 0.028 | 56 | 1.00 |

## epigenomics_1000@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.07756 ± 0.05 | 0.759 ± 0.064 | 31.78 ± 16 | 1.982 ± 0.17 | 0.2714 ± 0.079 | 30 | 1.00 |

## epigenomics_1000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.1262 ± 0.07 | 0.73 ± 0.054 | 42.6 ± 20 | 1.998 ± 0.17 | 0.2342 ± 0.064 | 36 | 1.00 |

## epigenomics_1000@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.1037 ± 0.039 | 0.7265 ± 0.053 | 69.89 ± 32 | 1.945 ± 0.19 | 0.2291 ± 0.062 | 57 | 1.00 |

## inspiral_100@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.03634 ± 0.026 | 0.8587 ± 0.039 | 30.12 ± 13 | 1.713 ± 0.057 | 0.6062 ± 0.041 | 56 | 1.00 |

## inspiral_100@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.09738 ± 0.083 | 0.897 ± 0.066 | 28.41 ± 10 | 1.561 ± 0.14 | 0.6828 ± 0.07 | 63 | 1.00 |

## inspiral_100@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.09234 ± 0.037 | 0.8357 ± 0.039 | 50.54 ± 14 | 1.621 ± 0.056 | 0.5264 ± 0.03 | 72 | 1.00 |

## inspiral_400@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.06474 ± 0.081 | 0.7942 ± 0.035 | 54.18 ± 35 | 1.95 ± 0.051 | 0.3489 ± 0.033 | 55 | 1.00 |

## inspiral_400@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.09051 ± 0.053 | 0.7401 ± 0.031 | 58.43 ± 26 | 2.023 ± 0.1 | 0.2389 ± 0.038 | 57 | 1.00 |

## inspiral_400@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.09614 ± 0.048 | 0.7321 ± 0.033 | 74.89 ± 32 | 1.967 ± 0.11 | 0.2425 ± 0.03 | 61 | 1.00 |

## inspiral_1000@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.08499 ± 0.075 | 0.7902 ± 0.071 | 36.59 ± 18 | 1.857 ± 0.19 | 0.3324 ± 0.082 | 36 | 1.00 |

## inspiral_1000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.1081 ± 0.054 | 0.7095 ± 0.056 | 51.35 ± 28 | 2.077 ± 0.18 | 0.2061 ± 0.062 | 42 | 1.00 |

## inspiral_1000@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.08734 ± 0.034 | 0.702 ± 0.028 | 74.02 ± 31 | 2.172 ± 0.12 | 0.1463 ± 0.028 | 50 | 1.00 |

## montage_100@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.05187 ± 0.03 | 0.7641 ± 0.037 | 33.84 ± 15 | 2.036 ± 0.078 | 0.4342 ± 0.043 | 52 | 1.00 |

## montage_100@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.1198 ± 0.087 | 0.7955 ± 0.067 | 40.02 ± 10 | 1.88 ± 0.17 | 0.5304 ± 0.069 | 74 | 1.00 |

## montage_100@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.08479 ± 0.053 | 0.874 ± 0.056 | 39.32 ± 8.3 | 1.507 ± 0.13 | 0.625 ± 0.069 | 71 | 1.00 |

## montage_400@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.08183 ± 0.073 | 0.7393 ± 0.024 | 56.06 ± 19 | 2.123 ± 0.031 | 0.2699 ± 0.025 | 64 | 1.00 |

## montage_400@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.04637 ± 0.032 | 0.7465 ± 0.022 | 56.87 ± 24 | 2.069 ± 0.068 | 0.2558 ± 0.024 | 59 | 1.00 |

## montage_400@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.06358 ± 0.035 | 0.7573 ± 0.028 | 56.41 ± 18 | 1.912 ± 0.063 | 0.287 ± 0.029 | 56 | 1.00 |

## montage_1000@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.08979 ± 0.065 | 0.7745 ± 0.085 | 27.52 ± 13 | 1.923 ± 0.23 | 0.3089 ± 0.11 | 30 | 1.00 |

## montage_1000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.1355 ± 0.067 | 0.7311 ± 0.082 | 42.27 ± 23 | 1.948 ± 0.23 | 0.2753 ± 0.094 | 39 | 1.00 |

## montage_1000@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.1084 ± 0.055 | 0.7418 ± 0.061 | 57.26 ± 25 | 1.892 ± 0.19 | 0.2506 ± 0.071 | 49 | 1.00 |

## sipht_100@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.05478 ± 0.051 | 0.8022 ± 0.065 | 36.54 ± 10 | 1.92 ± 0.15 | 0.4734 ± 0.068 | 56 | 1.00 |

## sipht_100@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.06253 ± 0.03 | 0.8552 ± 0.051 | 39.83 ± 7.4 | 1.638 ± 0.11 | 0.6048 ± 0.051 | 80 | 1.00 |

## sipht_100@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.06901 ± 0.036 | 0.878 ± 0.042 | 39.59 ± 13 | 1.469 ± 0.072 | 0.5942 ± 0.037 | 57 | 1.00 |

## sipht_400@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.05334 ± 0.049 | 0.7529 ± 0.032 | 47.26 ± 22 | 2.057 ± 0.033 | 0.2927 ± 0.024 | 52 | 1.00 |

## sipht_400@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.06725 ± 0.045 | 0.77 ± 0.026 | 49.97 ± 20 | 1.909 ± 0.049 | 0.3073 ± 0.025 | 51 | 1.00 |

## sipht_400@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.04394 ± 0.021 | 0.7446 ± 0.013 | 71.48 ± 29 | 2.011 ± 0.047 | 0.235 ± 0.01 | 58 | 1.00 |

## sipht_1000@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.08736 ± 0.044 | 0.7764 ± 0.065 | 32.89 ± 15 | 1.95 ± 0.18 | 0.3073 ± 0.081 | 34 | 1.00 |

## sipht_1000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.0836 ± 0.035 | 0.7372 ± 0.039 | 36.99 ± 23 | 2.026 ± 0.15 | 0.2089 ± 0.046 | 28 | 1.00 |

## sipht_1000@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| imoea_dqn | 0.06541 ± 0.037 | 0.697 ± 0.027 | 61.89 ± 25 | 2.29 ± 0.13 | 0.1164 ± 0.024 | 39 | 1.00 |
