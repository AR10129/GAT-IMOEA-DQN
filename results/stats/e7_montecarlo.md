# E7 Monte Carlo: `e7`

200 crisp scenarios per instance, every interval input drawn uniformly and independently; the same scenarios for every policy. HV (ref 1.1) and IGD on the E3 normalisation; IGD against each scenario's pooled crisp front. `final better` = share of scenarios where final's seed-mean beats the policy's.

## medium_250_i0@x20

Solutions: 9832. Containment (all three objectives inside the predicted interval): **100.00%** of solution-scenario pairs. Mean position inside the interval (makespan / energy / cost): 0.45 / 0.48 / 0.47. Crisp 2.5-97.5% spread as a share of the interval width: 0.09 / 0.10 / 0.91. Kendall tau, E3 interval IGD order vs Monte Carlo IGD order: 0.91.

| Policy | MC HV mean ± sd | MC IGD mean ± sd | final better (HV) | final better (IGD) | E3 IGD |
|---|---|---|---|---|---|
| Ours (final) | 0.8526 ± 0.0372 | 0.0123 ± 0.0004 | – | – | 0.0114 |
| IMOEA/D | 0.8262 ± 0.0387 | 0.0308 ± 0.0010 | 100% | 100% | 0.0343 |
| IMOEA/D-AWN | 0.8171 ± 0.0382 | 0.0368 ± 0.0010 | 100% | 100% | 0.0403 |
| IMEA-IED | 0.6378 ± 0.0322 | 0.1381 ± 0.0054 | 100% | 100% | 0.1531 |
| IP-MOEA | 0.6574 ± 0.0367 | 0.1432 ± 0.0081 | 100% | 100% | 0.1781 |
| NSGA-II (mid) | 0.7654 ± 0.0358 | 0.0623 ± 0.0017 | 100% | 100% | 0.0689 |
| NSGA-II (int) | 0.7663 ± 0.0358 | 0.0625 ± 0.0016 | 100% | 100% | 0.0689 |
| NSGA-III (mid) | 0.6504 ± 0.0444 | 0.1807 ± 0.0089 | 100% | 100% | 0.2340 |
| Jin backbone (fixed) | 0.5912 ± 0.0282 | 0.1574 ± 0.0064 | 100% | 100% | 0.1620 |
| IMOEA-DQN | 0.6088 ± 0.0473 | 0.2356 ± 0.0078 | 100% | 100% | 0.2904 |

## montage_400@x20

Solutions: 9719. Containment (all three objectives inside the predicted interval): **100.00%** of solution-scenario pairs. Mean position inside the interval (makespan / energy / cost): 0.44 / 0.47 / 0.46. Crisp 2.5-97.5% spread as a share of the interval width: 0.44 / 0.47 / 0.34. Kendall tau, E3 interval IGD order vs Monte Carlo IGD order: 0.96.

| Policy | MC HV mean ± sd | MC IGD mean ± sd | final better (HV) | final better (IGD) | E3 IGD |
|---|---|---|---|---|---|
| Ours (final) | 0.9068 ± 0.0375 | 0.0118 ± 0.0005 | – | – | 0.0107 |
| IMOEA/D | 0.8926 ± 0.0380 | 0.0103 ± 0.0004 | 100% | 0% | 0.0117 |
| IMOEA/D-AWN | 0.8887 ± 0.0377 | 0.0134 ± 0.0005 | 100% | 100% | 0.0135 |
| IMEA-IED | 0.6793 ± 0.0491 | 0.1519 ± 0.0063 | 100% | 100% | 0.2013 |
| IP-MOEA | 0.7227 ± 0.0441 | 0.0934 ± 0.0038 | 100% | 100% | 0.1314 |
| NSGA-II (mid) | 0.8253 ± 0.0409 | 0.0323 ± 0.0012 | 100% | 100% | 0.0460 |
| NSGA-II (int) | 0.8224 ± 0.0410 | 0.0335 ± 0.0013 | 100% | 100% | 0.0480 |
| NSGA-III (mid) | 0.7139 ± 0.0473 | 0.1158 ± 0.0048 | 100% | 100% | 0.1580 |
| Jin backbone (fixed) | 0.6990 ± 0.0410 | 0.0882 ± 0.0033 | 100% | 100% | 0.1266 |
| IMOEA-DQN | 0.6676 ± 0.0481 | 0.1563 ± 0.0058 | 100% | 100% | 0.2027 |

## cybershake_1000@x20

Solutions: 9367. Containment (all three objectives inside the predicted interval): **100.00%** of solution-scenario pairs. Mean position inside the interval (makespan / energy / cost): 0.45 / 0.46 / 0.46. Crisp 2.5-97.5% spread as a share of the interval width: 0.40 / 0.36 / 0.29. Kendall tau, E3 interval IGD order vs Monte Carlo IGD order: 1.00.

| Policy | MC HV mean ± sd | MC IGD mean ± sd | final better (HV) | final better (IGD) | E3 IGD |
|---|---|---|---|---|---|
| Ours (final) | 1.0071 ± 0.0307 | 0.0238 ± 0.0012 | – | – | 0.0276 |
| IMOEA/D | 0.9995 ± 0.0293 | 0.0337 ± 0.0017 | 100% | 100% | 0.0321 |
| IMOEA/D-AWN | 0.9427 ± 0.0266 | 0.0896 ± 0.0056 | 100% | 100% | 0.0838 |
| IMEA-IED | 0.8713 ± 0.0357 | 0.1392 ± 0.0098 | 100% | 100% | 0.1859 |
| IP-MOEA | 0.8631 ± 0.0353 | 0.1285 ± 0.0096 | 100% | 100% | 0.1739 |
| NSGA-II (mid) | 0.8936 ± 0.0325 | 0.0970 ± 0.0064 | 100% | 100% | 0.1216 |
| NSGA-II (int) | 0.8937 ± 0.0324 | 0.0951 ± 0.0062 | 100% | 100% | 0.1179 |
| NSGA-III (mid) | 0.8822 ± 0.0367 | 0.1358 ± 0.0102 | 100% | 100% | 0.1830 |
| Jin backbone (fixed) | 0.8273 ± 0.0325 | 0.1232 ± 0.0080 | 100% | 100% | 0.1586 |
| IMOEA-DQN | 0.7647 ± 0.0318 | 0.2341 ± 0.0090 | 100% | 100% | 0.2852 |

## epigenomics_1000@x20

Solutions: 9221. Containment (all three objectives inside the predicted interval): **100.00%** of solution-scenario pairs. Mean position inside the interval (makespan / energy / cost): 0.44 / 0.47 / 0.46. Crisp 2.5-97.5% spread as a share of the interval width: 0.41 / 0.45 / 0.30. Kendall tau, E3 interval IGD order vs Monte Carlo IGD order: 0.91.

| Policy | MC HV mean ± sd | MC IGD mean ± sd | final better (HV) | final better (IGD) | E3 IGD |
|---|---|---|---|---|---|
| Ours (final) | 0.9905 ± 0.0320 | 0.0290 ± 0.0017 | – | – | 0.0300 |
| IMOEA/D | 0.9952 ± 0.0303 | 0.0396 ± 0.0021 | 10% | 100% | 0.0383 |
| IMOEA/D-AWN | 0.9281 ± 0.0269 | 0.1081 ± 0.0066 | 100% | 100% | 0.1016 |
| IMEA-IED | 0.8662 ± 0.0390 | 0.1395 ± 0.0096 | 100% | 100% | 0.1924 |
| IP-MOEA | 0.8631 ± 0.0373 | 0.1174 ± 0.0087 | 100% | 100% | 0.1633 |
| NSGA-II (mid) | 0.8867 ± 0.0349 | 0.0994 ± 0.0065 | 100% | 100% | 0.1258 |
| NSGA-II (int) | 0.8847 ± 0.0346 | 0.1005 ± 0.0064 | 100% | 100% | 0.1261 |
| NSGA-III (mid) | 0.8737 ± 0.0394 | 0.1368 ± 0.0097 | 100% | 100% | 0.1899 |
| Jin backbone (fixed) | 0.8281 ± 0.0353 | 0.1271 ± 0.0083 | 100% | 100% | 0.1678 |
| IMOEA-DQN | 0.7504 ± 0.0332 | 0.2222 ± 0.0090 | 100% | 100% | 0.2735 |
