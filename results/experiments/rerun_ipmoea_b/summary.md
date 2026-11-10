# Experiment `rerun_ipmoea_b`

One shared normalization per workflow across all policies below. IGD ↓ · HV_jin ↑ · Imprecision ↓ · PIBI ↓ · HV_hi ↑. Agent policies: each eval seed is averaged over training seeds. Symbols: Wilcoxon signed-rank test (paired by seed, 5%) against *ip_moea_hv* — `+` better, `−` worse, `=` no significant difference.

## Overall (mean over workflows; rank by IGD)

| Policy | mean rank | IGD | HV_jin | Imprecision | PIBI | HV_hi | +/−/= vs ip_moea_hv (IGD, HV_jin, Imprecision) |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 1.00 | 0.01602 | 0.7243 | 123.9 | 2.107 | 0.2899 | (reference) |

## cybershake_100@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.005129 ± 0.0014 | 0.7819 ± 0.011 | 78.33 ± 2 | 1.994 ± 0.0042 | 0.5018 ± 0.0071 | 104 | 1.00 |

## cybershake_100@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01073 ± 0.0035 | 0.7921 ± 0.011 | 81.14 ± 2.7 | 1.917 ± 0.0051 | 0.5078 ± 0.0081 | 104 | 1.00 |

## cybershake_100@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01923 ± 0.003 | 0.8116 ± 0.011 | 89.73 ± 2 | 1.792 ± 0.0077 | 0.5097 ± 0.0089 | 104 | 1.00 |

## cybershake_400@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.00608 ± 0.0037 | 0.7077 ± 0.015 | 118.4 ± 3.6 | 2.211 ± 0.007 | 0.2627 ± 0.0059 | 105 | 1.00 |

## cybershake_400@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.008868 ± 0.0016 | 0.7156 ± 0.019 | 125.4 ± 3.5 | 2.112 ± 0.013 | 0.2732 ± 0.0079 | 104 | 1.00 |

## cybershake_400@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.02057 ± 0.008 | 0.7492 ± 0.016 | 123.2 ± 2.4 | 1.971 ± 0.013 | 0.317 ± 0.0072 | 105 | 1.00 |

## cybershake_1000@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.03821 ± 0.018 | 0.6423 ± 0.037 | 131.2 ± 9.6 | 2.229 ± 0.046 | 0.178 ± 0.013 | 100 | 1.00 |

## cybershake_1000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.0187 ± 0.0056 | 0.682 ± 0.027 | 146.6 ± 4.5 | 2.148 ± 0.037 | 0.171 ± 0.0087 | 104 | 1.00 |

## cybershake_1000@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.0299 ± 0.011 | 0.7049 ± 0.021 | 144.4 ± 4.4 | 2.083 ± 0.034 | 0.1982 ± 0.007 | 104 | 1.00 |

## epigenomics_100@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.006024 ± 0.0027 | 0.7594 ± 0.014 | 83.18 ± 2.2 | 2.066 ± 0.0048 | 0.4533 ± 0.0061 | 104 | 1.00 |

## epigenomics_100@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01317 ± 0.0026 | 0.753 ± 0.021 | 89.63 ± 1.9 | 1.974 ± 0.0084 | 0.4462 ± 0.011 | 104 | 1.00 |

## epigenomics_100@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.03506 ± 0.018 | 0.8177 ± 0.013 | 98.99 ± 4.3 | 1.647 ± 0.035 | 0.489 ± 0.019 | 104 | 1.00 |

## epigenomics_400@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.005873 ± 0.002 | 0.7304 ± 0.016 | 118.3 ± 3.8 | 2.129 ± 0.0081 | 0.288 ± 0.0074 | 104 | 1.00 |

## epigenomics_400@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01468 ± 0.003 | 0.6971 ± 0.028 | 128.8 ± 3.9 | 2.196 ± 0.021 | 0.2368 ± 0.01 | 104 | 1.00 |

## epigenomics_400@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.02736 ± 0.0049 | 0.7161 ± 0.012 | 145.7 ± 3.8 | 2.079 ± 0.022 | 0.2377 ± 0.0085 | 104 | 1.00 |

## epigenomics_1000@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01605 ± 0.011 | 0.6764 ± 0.04 | 136.9 ± 11 | 2.195 ± 0.059 | 0.1706 ± 0.015 | 101 | 1.00 |

## epigenomics_1000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01668 ± 0.005 | 0.6683 ± 0.037 | 150.2 ± 4.5 | 2.204 ± 0.071 | 0.1569 ± 0.013 | 104 | 1.00 |

## epigenomics_1000@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.04126 ± 0.0058 | 0.6856 ± 0.029 | 158.2 ± 6.1 | 2.147 ± 0.069 | 0.1715 ± 0.013 | 104 | 1.00 |

## inspiral_100@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.00592 ± 0.0037 | 0.8142 ± 0.0098 | 79.29 ± 2.8 | 1.931 ± 0.0047 | 0.5302 ± 0.0067 | 104 | 1.00 |

## inspiral_100@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.007215 ± 0.0015 | 0.7739 ± 0.011 | 95.18 ± 1.3 | 2.024 ± 0.0044 | 0.4273 ± 0.0065 | 104 | 1.00 |

## inspiral_100@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01466 ± 0.004 | 0.7954 ± 0.014 | 104.9 ± 4.2 | 1.839 ± 0.0074 | 0.4322 ± 0.0072 | 102 | 1.00 |

## inspiral_400@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.005773 ± 0.0021 | 0.7152 ± 0.02 | 117.5 ± 4.2 | 2.187 ± 0.011 | 0.278 ± 0.0089 | 104 | 1.00 |

## inspiral_400@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01559 ± 0.011 | 0.697 ± 0.019 | 125.2 ± 7 | 2.207 ± 0.015 | 0.2396 ± 0.0072 | 104 | 1.00 |

## inspiral_400@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01955 ± 0.007 | 0.7026 ± 0.0051 | 155.2 ± 4.9 | 2.257 ± 0.0097 | 0.1831 ± 0.0028 | 103 | 1.00 |

## inspiral_1000@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.02031 ± 0.012 | 0.6646 ± 0.028 | 134.2 ± 9 | 2.216 ± 0.036 | 0.1666 ± 0.0082 | 99 | 1.00 |

## inspiral_1000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01565 ± 0.0058 | 0.6772 ± 0.025 | 151.5 ± 5.7 | 2.214 ± 0.027 | 0.1568 ± 0.0073 | 103 | 1.00 |

## inspiral_1000@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.02878 ± 0.012 | 0.6692 ± 0.017 | 182.9 ± 4.7 | 2.405 ± 0.033 | 0.1035 ± 0.006 | 104 | 1.00 |

## montage_100@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.00698 ± 0.0038 | 0.7486 ± 0.0086 | 84.38 ± 3 | 2.095 ± 0.0048 | 0.436 ± 0.0056 | 105 | 1.00 |

## montage_100@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01093 ± 0.0061 | 0.7743 ± 0.017 | 91.88 ± 3.9 | 2.01 ± 0.0077 | 0.4325 ± 0.0091 | 104 | 1.00 |

## montage_100@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.02347 ± 0.0054 | 0.7639 ± 0.016 | 102.5 ± 3.8 | 1.869 ± 0.014 | 0.4174 ± 0.01 | 104 | 1.00 |

## montage_400@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.008108 ± 0.0033 | 0.7069 ± 0.034 | 120 ± 5.2 | 2.182 ± 0.023 | 0.2642 ± 0.015 | 104 | 1.00 |

## montage_400@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01 ± 0.004 | 0.7177 ± 0.021 | 121.6 ± 4.1 | 2.121 ± 0.011 | 0.2697 ± 0.0098 | 104 | 1.00 |

## montage_400@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01908 ± 0.003 | 0.7223 ± 0.011 | 141.4 ± 3 | 2.074 ± 0.01 | 0.2397 ± 0.0044 | 105 | 1.00 |

## montage_1000@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01616 ± 0.012 | 0.6873 ± 0.045 | 133.3 ± 7.8 | 2.185 ± 0.069 | 0.1779 ± 0.016 | 103 | 1.00 |

## montage_1000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01246 ± 0.0023 | 0.6942 ± 0.035 | 144.2 ± 6.4 | 2.137 ± 0.055 | 0.1724 ± 0.013 | 104 | 1.00 |

## montage_1000@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.03093 ± 0.013 | 0.6847 ± 0.019 | 159.9 ± 4.8 | 2.167 ± 0.036 | 0.1565 ± 0.0059 | 104 | 1.00 |

## sipht_100@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.00518 ± 0.0014 | 0.7807 ± 0.014 | 85.55 ± 1.8 | 1.997 ± 0.0059 | 0.469 ± 0.0079 | 104 | 1.00 |

## sipht_100@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.005198 ± 0.00074 | 0.8026 ± 0.019 | 97.42 ± 1.9 | 1.922 ± 0.007 | 0.4628 ± 0.015 | 105 | 1.00 |

## sipht_100@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01061 ± 0.0038 | 0.7996 ± 0.011 | 116.3 ± 3.3 | 1.907 ± 0.0035 | 0.3991 ± 0.0038 | 101 | 1.00 |

## sipht_400@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.005958 ± 0.0032 | 0.7264 ± 0.0099 | 115.3 ± 3.3 | 2.185 ± 0.0054 | 0.2804 ± 0.0045 | 104 | 1.00 |

## sipht_400@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.008934 ± 0.0058 | 0.7012 ± 0.018 | 130.3 ± 4.1 | 2.189 ± 0.012 | 0.2313 ± 0.0073 | 104 | 1.00 |

## sipht_400@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01018 ± 0.0034 | 0.697 ± 0.01 | 163 ± 3.5 | 2.325 ± 0.0081 | 0.162 ± 0.0049 | 101 | 1.00 |

## sipht_1000@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.03007 ± 0.015 | 0.6459 ± 0.048 | 127.6 ± 8.6 | 2.258 ± 0.067 | 0.1745 ± 0.016 | 103 | 1.00 |

## sipht_1000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.01654 ± 0.011 | 0.6827 ± 0.019 | 153.6 ± 6.7 | 2.213 ± 0.024 | 0.1485 ± 0.0064 | 103 | 1.00 |

## sipht_1000@x100

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| ip_moea_hv | 0.02327 ± 0.01 | 0.6575 ± 0.026 | 194.9 ± 6.2 | 2.587 ± 0.069 | 0.06366 ± 0.0069 | 102 | 1.00 |
