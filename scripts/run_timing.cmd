@echo off
rem B4 clean timing: one run at a time, nothing else running, laptop awake
cd /d %~dp0..
.venv\Scripts\python.exe -u scripts\experiment.py --name timing_a --policies final imoead_orig imoead_awn imea_ied ip_moea nsga2_mid nsga2_int nsga3_mid fixed_fav trained_stable --scenario default --seeds 2 --train-seeds 0 --energy-scope device --external-io device --makespan sum_apps --reference final --rank-metric IGD --generations 1000 --workers 1 --test-set small_20_i0@x20 medium_250_i0@x20 >> results\experiment_timing.log 2>&1
echo timing_a exit %errorlevel% >> results\experiment_timing.log 2>&1
.venv\Scripts\python.exe -u scripts\experiment.py --name timing_b --policies final imoead_orig imoead_awn imea_ied ip_moea nsga2_mid nsga2_int nsga3_mid fixed_fav trained_stable --scenario jin --seeds 2 --train-seeds 0 --energy-scope device --external-io device --makespan sum_apps --reference final --rank-metric IGD --generations 1000 --workers 1 --test-set cybershake_100@x20 cybershake_400@x20 cybershake_1000@x20 cybershake_2000@x20 >> results\experiment_timing.log 2>&1
echo exit %errorlevel% >> results\experiment_timing.log 2>&1
