@echo off
rem E2 Suite A rerun (the first attempt hit a numba cache race at start-up); resumable
cd /d %~dp0..
.venv\Scripts\python.exe -u scripts\experiment.py --name suite_a_e2 --policies final imoead_orig fixed_fav --seeds 30 --scenario default --energy-scope device --external-io device --makespan sum_apps --reference final --rank-metric IGD --generations 1000 --workers 12 --test-set small_20_i0@x1 small_20_i0@x20 small_50_i0@x1 small_50_i0@x20 basic_60_i0@x1 basic_60_i0@x20 basic_100_i0@x1 basic_100_i0@x20 medium_100_i0@x1 medium_100_i0@x20 medium_250_i0@x1 medium_250_i0@x20 >> results\experiment_e2a.log 2>&1
echo exit %errorlevel% >> results\experiment_e2a.log 2>&1
