@echo off
rem Extreme-coverage pilot (final vs Rank C normalisation / axis-weight variants).
cd /d %~dp0..
.venv\Scripts\python.exe -u scripts\experiment.py --name pilot_ext --policies final imoead_orig final_rb final_ax final_rbax imoead_noadapt --seeds 5 --scenario jin --energy-scope device --external-io device --makespan sum_apps --reference final --rank-metric IGD --generations 1000 --workers 12 --test-set montage_400@x20 sipht_400@x20 cybershake_1000@x1 epigenomics_1000@x20 sipht_1000@x20 cybershake_2000@x100 epigenomics_2000@x20 inspiral_2000@x20 >> results\experiment_pilot.log 2>&1
echo exit %errorlevel% >> results\experiment_pilot.log
