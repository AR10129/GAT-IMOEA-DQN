@echo off
rem B4 timing for the corrected baselines (ip_moea_hv, imoea_dqn); final again as a control for machine drift.
rem One run at a time, nothing else running, laptop awake.
cd /d %~dp0..
.venv\Scripts\python.exe -u scripts\experiment.py --name timing2_a --policies final ip_moea_hv imoea_dqn --scenario default --seeds 2 --train-seeds 0 --energy-scope device --external-io device --makespan sum_apps --reference final --rank-metric IGD --generations 1000 --workers 1 --test-set small_20_i0@x20 medium_250_i0@x20 >> results\experiment_timing2.log 2>&1
echo timing2_a exit %errorlevel% >> results\experiment_timing2.log 2>&1
.venv\Scripts\python.exe -u scripts\experiment.py --name timing2_b --policies final ip_moea_hv imoea_dqn --scenario jin --seeds 2 --train-seeds 0 --energy-scope device --external-io device --makespan sum_apps --reference final --rank-metric IGD --generations 1000 --workers 1 --test-set cybershake_100@x20 cybershake_400@x20 cybershake_1000@x20 cybershake_2000@x20 >> results\experiment_timing2.log 2>&1
echo exit %errorlevel% >> results\experiment_timing2.log 2>&1
