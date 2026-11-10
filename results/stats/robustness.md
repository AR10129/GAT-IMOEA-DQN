# Normalisation robustness: final vs IMOEA/D

Each pool sets the per-instance normalisation and the IGD reference front. Symbol = final vs IMOEA/D, Wilcoxon rank-sum, two-sided, alpha 0.05 (`+` final better, `−` worse, `=` no significant difference); final 30 seeds, IMOEA/D 30 seeds.

Pools: `paper10` = final, imoead_orig, imoead_awn, imea_ied, ip_moea_hv, nsga2_mid, nsga2_int, nsga3_mid, fixed_fav, imoea_dqn; `union12` = final, imoead_orig, imoead_awn, imea_ied, ip_moea_hv, nsga2_mid, nsga2_int, nsga3_mid, fixed_fav, imoea_dqn, ip_moea, trained_stable; `strong5` = final, imoead_orig, imoead_awn, nsga2_mid, nsga2_int; `pair2` = final, imoead_orig

## Suite A (12 instances)

| Pool | IGD +/−/= | HV_jin +/−/= |
|---|---|---|
| paper10 | 12/0/0 | 12/0/0 |
| union12 | 12/0/0 | 12/0/0 |
| strong5 | 12/0/0 | 12/0/0 |
| pair2 | 12/0/0 | 12/0/0 |

Instances where the verdict is not `+` in every pool (IGD final / IMOEA/D, symbol):

| Instance | paper10 | union12 | strong5 | pair2 | HV_jin symbols |
|---|---|---|---|---|---|

## Suite B (45 instances)

| Pool | IGD +/−/= | HV_jin +/−/= |
|---|---|---|
| paper10 | 43/1/1 | 42/2/1 |
| union12 | 42/1/2 | 42/2/1 |
| strong5 | 33/9/3 | 42/2/1 |
| pair2 | 34/6/5 | 42/1/2 |

Instances where the verdict is not `+` in every pool (IGD final / IMOEA/D, symbol):

| Instance | paper10 | union12 | strong5 | pair2 | HV_jin symbols |
|---|---|---|---|---|---|
| cybershake_1000@x1 | 0.0276 / 0.0301 + | 0.0267 / 0.0276 = | 0.0318 / 0.0242 - | 0.0307 / 0.0243 - | + + + + |
| cybershake_1000@x20 | 0.0276 / 0.0321 + | 0.0271 / 0.0299 + | 0.0313 / 0.0254 - | 0.0287 / 0.0260 - | + + + + |
| epigenomics_1000@x1 | 0.0258 / 0.0335 + | 0.0253 / 0.0312 + | 0.0321 / 0.0245 - | 0.0301 / 0.0253 - | + + + + |
| epigenomics_1000@x20 | 0.0300 / 0.0383 + | 0.0290 / 0.0345 + | 0.0400 / 0.0254 - | 0.0378 / 0.0271 - | + + + + |
| inspiral_1000@x1 | 0.0255 / 0.0298 + | 0.0247 / 0.0273 + | 0.0278 / 0.0228 - | 0.0252 / 0.0236 = | + + + + |
| inspiral_1000@x20 | 0.0328 / 0.0388 + | 0.0304 / 0.0337 + | 0.0379 / 0.0220 - | 0.0380 / 0.0249 - | = = = = |
| inspiral_1000@x100 | 0.0497 / 0.0435 - | 0.0447 / 0.0341 - | 0.0437 / 0.0344 - | 0.0370 / 0.0354 = | - - - - |
| montage_100@x100 | 0.0258 / 0.0254 = | 0.0218 / 0.0216 = | 0.0254 / 0.0258 = | 0.0194 / 0.0223 + | + + + + |
| montage_1000@x1 | 0.0148 / 0.0381 + | 0.0143 / 0.0345 + | 0.0239 / 0.0227 = | 0.0223 / 0.0224 = | + + + + |
| montage_1000@x20 | 0.0218 / 0.0481 + | 0.0210 / 0.0442 + | 0.0309 / 0.0289 - | 0.0312 / 0.0325 = | + + + + |
| sipht_1000@x1 | 0.0156 / 0.0327 + | 0.0151 / 0.0297 + | 0.0234 / 0.0230 = | 0.0227 / 0.0236 = | + + + + |
| sipht_1000@x20 | 0.0248 / 0.0393 + | 0.0229 / 0.0338 + | 0.0332 / 0.0231 - | 0.0335 / 0.0261 - | - - - = |
