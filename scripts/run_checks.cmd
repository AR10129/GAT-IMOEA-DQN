@echo off
rem Checks: IP-MOEA with hypervolume-contribution crowding, IMOEA-DQN as published (online, MLP); resumable
cd /d %~dp0..
.venv\Scripts\python.exe -u scripts\experiment.py --name checks_a --scenario default --policies ip_moea_hv imoea_dqn --seeds 10 --train-seeds 0 --energy-scope device --external-io device --makespan sum_apps --reference ip_moea_hv --rank-metric IGD --generations 1000 --workers 12 --test-set medium_250_i0@x20 >> results\experiment_checks.log 2>&1
echo checks_a exit %errorlevel% >> results\experiment_checks.log 2>&1
.venv\Scripts\python.exe -u scripts\experiment.py --name checks_b --scenario jin --policies ip_moea_hv imoea_dqn --seeds 10 --train-seeds 0 --energy-scope device --external-io device --makespan sum_apps --reference ip_moea_hv --rank-metric IGD --generations 1000 --workers 12 --test-set montage_400@x20 cybershake_1000@x20 epigenomics_1000@x20 >> results\experiment_checks.log 2>&1
echo exit %errorlevel% >> results\experiment_checks.log 2>&1
