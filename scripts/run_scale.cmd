@echo off
rem scale: split from run_e.cmd; resumable (finished runs are skipped)
cd /d %~dp0..
.venv\Scripts\python.exe -u scripts\experiment.py --name scale_2000 --policies final imoead_orig fixed_fav nsga2_mid --seeds 10 --scenario jin --energy-scope device --external-io device --makespan sum_apps  --reference final --rank-metric IGD --generations 1000 --workers 12 --test-set cybershake_2000@x1 cybershake_2000@x20 cybershake_2000@x100 epigenomics_2000@x1 epigenomics_2000@x20 epigenomics_2000@x100 inspiral_2000@x1 inspiral_2000@x20 inspiral_2000@x100 montage_2000@x1 montage_2000@x20 montage_2000@x100 sipht_2000@x1 sipht_2000@x20 sipht_2000@x100 >> results\experiment_scale.log 2>&1
echo scale_2000 exit %errorlevel% >> results\experiment_scale.log 2>&1
echo exit %errorlevel% >> results\experiment_scale.log 2>&1
