@echo off
rem 1000-task investigation: why imoead_orig wins HV at 1000 tasks
cd /d %~dp0..
.venv\Scripts\python.exe -u scripts\experiment.py --name inv_act --policies final imoead_orig final_a23 final_a12 final_p09 --generations 1000 --reference final --rank-metric IGD --scenario jin --energy-scope device --external-io device --makespan sum_apps --seeds 5 --test-set cybershake_1000@x1 epigenomics_1000@x20 sipht_1000@x20 --workers 12 >> results\experiment_inv.log 2>&1
echo inv_act exit %errorlevel% >> results\experiment_inv.log
.venv\Scripts\python.exe -u scripts\experiment.py --name inv_g2000 --policies final imoead_orig --generations 2000 --reference final --rank-metric IGD --scenario jin --energy-scope device --external-io device --makespan sum_apps --seeds 5 --test-set cybershake_1000@x1 epigenomics_1000@x20 sipht_1000@x20 --workers 12 >> results\experiment_inv.log 2>&1
echo exit %errorlevel% >> results\experiment_inv.log
