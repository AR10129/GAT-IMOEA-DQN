# Experiment `timing2_b`

One shared normalization per workflow across all policies below. IGD ↓ · HV_jin ↑ · Imprecision ↓ · PIBI ↓ · HV_hi ↑. Agent policies: each eval seed is averaged over training seeds. Symbols: Wilcoxon signed-rank test (paired by seed, 5%) against *final* — `+` better, `−` worse, `=` no significant difference.

## Overall (mean over workflows; rank by IGD)

| Policy | mean rank | IGD | HV_jin | Imprecision | PIBI | HV_hi | +/−/= vs final (IGD, HV_jin, Imprecision) |
|---|---|---|---|---|---|---|---|
| final | 1.00 | 0.02401 | 0.8955 | 69.54 | 1.656 | 0.6294 | (reference) |
| ip_moea_hv | 2.00 | 0.09909 | 0.7789 | 58.91 | 1.689 | 0.5439 | 0/0/4 · 0/0/4 · 0/0/4 |
| imoea_dqn | 3.00 | 0.2697 | 0.6132 | 20.84 | 2.391 | 0.3144 | 0/0/4 · 0/0/4 · 0/0/4 |

## cybershake_100@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| final | 0.01183 ± 0.00027 | 0.8925 ± 0.002 | 51.91 ± 0.16 | 1.723 ± 0.0017 | 0.6962 ± 9.6e-05 | 90 | 1.00 |
| ip_moea_hv | 0.03829 ± 0.0051 = | 0.8047 ± 0.0057 = | 55.37 ± 1.4 = | 1.761 ± 0.0042 = | 0.6371 ± 0.005 = | 104 | 2.00 |
| imoea_dqn | 0.2375 ± 0.14 = | 0.6754 ± 0.083 = | 18.82 ± 7.1 = | 2.218 ± 0.33 = | 0.4842 ± 0.094 = | 49 | 3.00 |

## cybershake_400@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| final | 0.008853 ± 0.00024 | 0.8654 ± 0.0004 | 77.12 ± 1.5 | 1.759 ± 0.00012 | 0.594 ± 0.00075 | 102 | 1.00 |
| ip_moea_hv | 0.09478 ± 0.0036 = | 0.7042 ± 0.0039 = | 62.59 ± 1.5 = | 1.923 ± 0.0093 = | 0.4672 ± 0.0015 = | 104 | 2.00 |
| imoea_dqn | 0.1954 ± 0.019 = | 0.6368 ± 0.001 = | 12.11 ± 3.2 = | 2.273 ± 0.017 = | 0.3619 ± 0.0049 = | 22 | 3.00 |

## cybershake_1000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| final | 0.01829 ± 0.004 | 0.9207 ± 0.0018 | 74.56 ± 4.4 | 1.586 ± 5.4e-05 | 0.6377 ± 0.0022 | 98 | 1.00 |
| ip_moea_hv | 0.1285 ± 0.0094 = | 0.7813 ± 0.0052 = | 59.39 ± 0.41 = | 1.66 ± 0.013 = | 0.5212 ± 0.0068 = | 104 | 2.00 |
| imoea_dqn | 0.3333 ± 0.024 = | 0.575 ± 0.01 = | 19.97 ± 15 = | 2.565 ± 0.066 = | 0.2258 ± 0.013 = | 31 | 3.00 |

## cybershake_2000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| final | 0.05708 ± 0.0086 | 0.9032 ± 0.016 | 74.57 ± 1.4 | 1.559 ± 0.051 | 0.5899 ± 0.024 | 92 | 1.00 |
| ip_moea_hv | 0.1348 ± 0.0063 = | 0.8255 ± 0.024 = | 58.28 ± 3.8 = | 1.412 ± 0.013 = | 0.5499 ± 0.022 = | 98 | 2.00 |
| imoea_dqn | 0.3127 ± 0.0015 = | 0.5656 ± 0.024 = | 32.46 ± 3.3 = | 2.511 ± 0.0083 = | 0.1856 ± 0.019 = | 43 | 3.00 |
