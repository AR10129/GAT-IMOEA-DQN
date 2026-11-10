# E7 Monte Carlo: `e7`

200 crisp scenarios per instance, every interval input drawn uniformly and independently; the same scenarios for every policy. HV (ref 1.1) and IGD on the E3 normalisation; IGD against each scenario's pooled crisp front. `final better` = share of scenarios where final's seed-mean beats the policy's.

## medium_250_i0@x20

Solutions: 9740. Containment (all three objectives inside the predicted interval): **100.00%** of solution-scenario pairs. Mean position inside the interval (makespan / energy / cost): 0.45 / 0.48 / 0.47. Crisp 2.5-97.5% spread as a share of the interval width: 0.09 / 0.09 / 0.85. Kendall tau, E3 interval IGD order vs Monte Carlo IGD order: 0.96.

| Policy | MC HV mean ± sd | MC IGD mean ± sd | final better (HV) | final better (IGD) | E3 IGD |
|---|---|---|---|---|---|
| Ours (final) | 0.9486 ± 0.0291 | 0.0122 ± 0.0005 | – | – | 0.0107 |
| IMOEA/D | 0.9312 ± 0.0302 | 0.0298 ± 0.0009 | 100% | 100% | 0.0324 |
| IMOEA/D-AWN | 0.9230 ± 0.0298 | 0.0346 ± 0.0010 | 100% | 100% | 0.0375 |
| IMEA-IED | 0.7255 ± 0.0253 | 0.1471 ± 0.0045 | 100% | 100% | 0.1536 |
| IP-MOEA | 0.5231 ± 0.0101 | 0.3096 ± 0.0129 | 100% | 100% | 0.3088 |
| NSGA-II (mid) | 0.8621 ± 0.0280 | 0.0595 ± 0.0017 | 100% | 100% | 0.0652 |
| NSGA-II (int) | 0.8632 ± 0.0280 | 0.0593 ± 0.0016 | 100% | 100% | 0.0650 |
| NSGA-III (mid) | 0.7761 ± 0.0348 | 0.1927 ± 0.0077 | 100% | 100% | 0.2322 |
| Jin backbone (fixed) | 0.6694 ± 0.0222 | 0.1702 ± 0.0056 | 100% | 100% | 0.1651 |
| IMOEA-DQN (ours model) | 0.8226 ± 0.0336 | 0.1371 ± 0.0052 | 100% | 100% | 0.1616 |

## montage_400@x20

Solutions: 9529. Containment (all three objectives inside the predicted interval): **100.00%** of solution-scenario pairs. Mean position inside the interval (makespan / energy / cost): 0.44 / 0.47 / 0.46. Crisp 2.5-97.5% spread as a share of the interval width: 0.43 / 0.45 / 0.32. Kendall tau, E3 interval IGD order vs Monte Carlo IGD order: 0.91.

| Policy | MC HV mean ± sd | MC IGD mean ± sd | final better (HV) | final better (IGD) | E3 IGD |
|---|---|---|---|---|---|
| Ours (final) | 1.0800 ± 0.0221 | 0.0115 ± 0.0006 | – | – | 0.0097 |
| IMOEA/D | 1.0717 ± 0.0224 | 0.0103 ± 0.0004 | 100% | 0% | 0.0108 |
| IMOEA/D-AWN | 1.0690 ± 0.0222 | 0.0126 ± 0.0006 | 100% | 100% | 0.0119 |
| IMEA-IED | 0.9487 ± 0.0287 | 0.1590 ± 0.0070 | 100% | 100% | 0.1890 |
| IP-MOEA | 0.7360 ± 0.0119 | 0.2806 ± 0.0154 | 100% | 100% | 0.2898 |
| NSGA-II (mid) | 1.0323 ± 0.0240 | 0.0327 ± 0.0013 | 100% | 100% | 0.0419 |
| NSGA-II (int) | 1.0303 ± 0.0241 | 0.0343 ± 0.0014 | 100% | 100% | 0.0438 |
| NSGA-III (mid) | 0.9688 ± 0.0276 | 0.1215 ± 0.0054 | 100% | 100% | 0.1474 |
| Jin backbone (fixed) | 0.9202 ± 0.0242 | 0.0915 ± 0.0039 | 100% | 100% | 0.1158 |
| IMOEA-DQN (ours model) | 0.9810 ± 0.0256 | 0.0807 ± 0.0034 | 100% | 100% | 0.1227 |

## cybershake_1000@x20

Solutions: 8908. Containment (all three objectives inside the predicted interval): **100.00%** of solution-scenario pairs. Mean position inside the interval (makespan / energy / cost): 0.45 / 0.46 / 0.45. Crisp 2.5-97.5% spread as a share of the interval width: 0.40 / 0.35 / 0.27. Kendall tau, E3 interval IGD order vs Monte Carlo IGD order: 0.96.

| Policy | MC HV mean ± sd | MC IGD mean ± sd | final better (HV) | final better (IGD) | E3 IGD |
|---|---|---|---|---|---|
| Ours (final) | 1.0479 ± 0.0265 | 0.0279 ± 0.0016 | – | – | 0.0297 |
| IMOEA/D | 1.0405 ± 0.0253 | 0.0301 ± 0.0014 | 100% | 88% | 0.0289 |
| IMOEA/D-AWN | 0.9891 ± 0.0230 | 0.0752 ± 0.0046 | 100% | 100% | 0.0726 |
| IMEA-IED | 0.9266 ± 0.0309 | 0.1683 ± 0.0123 | 100% | 100% | 0.1958 |
| IP-MOEA | 0.6210 ± 0.0119 | 0.3103 ± 0.0163 | 100% | 100% | 0.3206 |
| NSGA-II (mid) | 0.9423 ± 0.0282 | 0.1148 ± 0.0082 | 100% | 100% | 0.1269 |
| NSGA-II (int) | 0.9426 ± 0.0281 | 0.1113 ± 0.0079 | 100% | 100% | 0.1220 |
| NSGA-III (mid) | 0.9371 ± 0.0317 | 0.1659 ± 0.0127 | 100% | 100% | 0.1941 |
| Jin backbone (fixed) | 0.8784 ± 0.0283 | 0.1446 ± 0.0103 | 100% | 100% | 0.1646 |
| IMOEA-DQN (ours model) | 0.8483 ± 0.0278 | 0.2406 ± 0.0119 | 100% | 100% | 0.2558 |

## epigenomics_1000@x20

Solutions: 8729. Containment (all three objectives inside the predicted interval): **100.00%** of solution-scenario pairs. Mean position inside the interval (makespan / energy / cost): 0.44 / 0.47 / 0.45. Crisp 2.5-97.5% spread as a share of the interval width: 0.40 / 0.43 / 0.28. Kendall tau, E3 interval IGD order vs Monte Carlo IGD order: 1.00.

| Policy | MC HV mean ± sd | MC IGD mean ± sd | final better (HV) | final better (IGD) | E3 IGD |
|---|---|---|---|---|---|
| Ours (final) | 1.0414 ± 0.0273 | 0.0385 ± 0.0023 | – | – | 0.0346 |
| IMOEA/D | 1.0442 ± 0.0258 | 0.0346 ± 0.0018 | 19% | 10% | 0.0314 |
| IMOEA/D-AWN | 0.9842 ± 0.0229 | 0.0870 ± 0.0055 | 100% | 100% | 0.0818 |
| IMEA-IED | 0.9364 ± 0.0331 | 0.2030 ± 0.0131 | 100% | 100% | 0.2378 |
| IP-MOEA | 0.6129 ± 0.0123 | 0.3253 ± 0.0168 | 100% | 100% | 0.3312 |
| NSGA-II (mid) | 0.9484 ± 0.0297 | 0.1411 ± 0.0091 | 100% | 100% | 0.1516 |
| NSGA-II (int) | 0.9463 ± 0.0295 | 0.1410 ± 0.0089 | 100% | 100% | 0.1509 |
| NSGA-III (mid) | 0.9433 ± 0.0335 | 0.2009 ± 0.0133 | 100% | 100% | 0.2360 |
| Jin backbone (fixed) | 0.8935 ± 0.0301 | 0.1837 ± 0.0115 | 100% | 100% | 0.2051 |
| IMOEA-DQN (ours model) | 0.8495 ± 0.0298 | 0.2734 ± 0.0133 | 100% | 100% | 0.2602 |
