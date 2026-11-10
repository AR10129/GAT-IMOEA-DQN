@echo off
rem E5/E7 refresh: corrected baselines (ip_moea_hv, imoea_dqn) traced and with saved fronts; resumable.
rem Runs are copied into e5_* and e7_* afterwards.
cd /d %~dp0..
.venv\Scripts\python.exe -u scripts\experiment.py --name e57fix_a --scenario default --policies ip_moea_hv imoea_dqn --seeds 10 --train-seeds 0 --energy-scope device --external-io device --makespan sum_apps --reference ip_moea_hv --rank-metric IGD --generations 1000 --workers 12 --trace --save-front --test-set medium_250_i0@x20 >> results\experiment_e57fix.log 2>&1
echo e57fix_a exit %errorlevel% >> results\experiment_e57fix.log 2>&1
.venv\Scripts\python.exe -u scripts\experiment.py --name e57fix_b --scenario jin --policies ip_moea_hv imoea_dqn --seeds 10 --train-seeds 0 --energy-scope device --external-io device --makespan sum_apps --reference ip_moea_hv --rank-metric IGD --generations 1000 --workers 12 --trace --save-front --test-set montage_400@x20 cybershake_1000@x20 epigenomics_1000@x20 >> results\experiment_e57fix.log 2>&1
echo exit %errorlevel% >> results\experiment_e57fix.log 2>&1
