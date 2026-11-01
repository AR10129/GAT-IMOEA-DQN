# A1b: neighbourhood-mating parameters tuned on the training instances

T × p_local grid, action (5/6, 1/6, 0), tch backbone, 36 training instances (i1–i3, CCR ×1 and ×20) × 2 seeds, G_max = 1000, N = 105. Test instances (i0) not used.

**Chosen: T = 10, p_local = 1.0** — mean IGD rank 2.44 of 8; lower mean IGD than the old default (T = 20, p_local = 0.9) on 35/36 instances.

| setting | mean IGD rank | mean HV rank | IGD ratio (mean / worst) | best on # instances |
|---|---|---|---|---|
| T = 10, p_local = 1.0 | 2.44 | 2.19 | 1.068 / 1.411 | 16 |
| T = 5, p_local = 1.0 | 2.64 | 2.58 | 1.062 / 1.247 | 9 |
| T = 5, p_local = 0.9 | 3.11 | 3.14 | 1.087 / 1.309 | 6 |
| T = 3, p_local = 0.9 | 3.94 | 4.75 | 1.141 / 1.400 | 1 |
| T = 10, p_local = 0.9 | 4.78 | 4.92 | 1.268 / 1.864 | 1 |
| T = 3, p_local = 1.0 | 4.97 | 5.06 | 1.350 / 1.932 | 2 |
| T = 20, p_local = 1.0 | 6.28 | 5.89 | 1.675 / 2.558 | 1 |
| T = 20, p_local = 0.9 | 7.83 | 7.47 | 2.190 / 4.343 | 0 |

## Mean IGD rank by group

| group | T3_p90 | T3_p100 | T5_p90 | T5_p100 | T10_p90 | T10_p100 | T20_p90 | T20_p100 |
|---|---|---|---|---|---|---|---|---|
| CCR x1 | 4.28 | 4.94 | 2.89 | 2.56 | 4.83 | 2.22 | 7.83 | 6.44 |
| CCR x20 | 3.61 | 5.00 | 3.33 | 2.72 | 4.72 | 2.67 | 7.83 | 6.11 |
| small_20 | 4.17 | 4.00 | 4.17 | 3.67 | 4.33 | 1.83 | 7.83 | 6.00 |
| small_50 | 6.17 | 3.67 | 5.00 | 3.83 | 3.50 | 2.83 | 7.33 | 3.67 |
| basic_60 | 2.67 | 5.17 | 3.33 | 1.50 | 5.33 | 3.00 | 8.00 | 7.00 |
| basic_100 | 3.50 | 5.33 | 2.67 | 2.00 | 5.33 | 2.17 | 7.83 | 7.17 |
| medium_100 | 3.50 | 6.00 | 1.67 | 2.67 | 5.00 | 2.33 | 8.00 | 6.83 |
| medium_250 | 3.67 | 5.67 | 1.83 | 2.17 | 5.17 | 2.50 | 8.00 | 7.00 |
