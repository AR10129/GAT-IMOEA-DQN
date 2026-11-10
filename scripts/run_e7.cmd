@echo off
rem E7 Monte Carlo: final fronts with genotypes (X, O); resumable
cd /d %~dp0..
.venv\Scripts\python.exe -u scripts\experiment.py --name e7_a --scenario default --policies final imoead_orig imoead_awn imea_ied ip_moea nsga2_mid nsga2_int nsga3_mid fixed_fav trained_stable --seeds 10 --train-seeds 0 --energy-scope device --external-io device --makespan sum_apps --reference final --rank-metric IGD --generations 1000 --workers 12 --save-front --test-set medium_250_i0@x20 >> results\experiment_e7.log 2>&1
echo e7_a exit %errorlevel% >> results\experiment_e7.log 2>&1
.venv\Scripts\python.exe -u scripts\experiment.py --name e7_b --scenario jin --policies final imoead_orig imoead_awn imea_ied ip_moea nsga2_mid nsga2_int nsga3_mid fixed_fav trained_stable --seeds 10 --train-seeds 0 --energy-scope device --external-io device --makespan sum_apps --reference final --rank-metric IGD --generations 1000 --workers 12 --save-front --test-set montage_400@x20 cybershake_1000@x20 epigenomics_1000@x20 >> results\experiment_e7.log 2>&1
echo exit %errorlevel% >> results\experiment_e7.log 2>&1
