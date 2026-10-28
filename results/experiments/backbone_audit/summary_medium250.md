# Backbone audit (G_max = 1000, N = 105, fixed actions, no learning)

IGD ↓ · HV_jin ↑ · Imprecision ↓ · front size; mean ± std over seeds; `+`/`−`/`=` Wilcoxon vs base (5%); with 4 seeds the smallest possible p is 0.125, so read the means.

## medium_250_i0@x20

| Variant | IGD | HV_jin | Imprecision | front |
|---|---|---|---|---|
| base | 0.1551 ± 0.0055 | 0.5462 ± 0.027 | 36.45 ± 1.7 | 90 |
| near_tie | 0.1474 ± 0.0073 = | 0.5242 ± 0.0095 = | 42.2 ± 0.57 = | 88 |
| tch_near | 0.01559 ± 0.001 = | 0.817 ± 0.0034 = | 37.1 ± 0.85 = | 88 |
| tch_conv_near | 0.01435 ± 0.0011 = | 0.8194 ± 0.0027 = | 41.79 ± 0.76 = | 102 |
| imoead | 0.03526 ± 0.0031 = | 0.7915 ± 0.0049 = | 40.69 ± 0.34 = | 102 |

