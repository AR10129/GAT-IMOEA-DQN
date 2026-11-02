@echo off
rem Suite B: 5 WfCommons/Pegasus families x 100/400/1000 tasks x data x1/x20/x100 (low/medium/high CCR),
rem Jin scenario, workflows rescaled into Jin ranges at load time; 10 seeds, Gmax 1000.
rem final = the frozen method; fixed_fav = Jin et al. backbone with the same action; heuristics = B0.
rem Re-running resumes: finished runs are skipped. Launch detached:
rem   powershell -Command "Start-Process cmd -ArgumentList '/c scripts\run_suite_b.cmd' -WindowStyle Hidden"
cd /d %~dp0..
.venv\Scripts\python.exe -u scripts\experiment.py --name suite_b --reuse-from suite_b_pilot --policies final imoead ip_moea fixed_fav heuristics --reference final --rank-metric IGD --scenario jin --energy-scope device --external-io device --makespan sum_apps --generations 1000 --seeds 10 --test-set cybershake_100@x1 cybershake_100@x20 cybershake_100@x100 cybershake_400@x1 cybershake_400@x20 cybershake_400@x100 cybershake_1000@x1 cybershake_1000@x20 cybershake_1000@x100 epigenomics_100@x1 epigenomics_100@x20 epigenomics_100@x100 epigenomics_400@x1 epigenomics_400@x20 epigenomics_400@x100 epigenomics_1000@x1 epigenomics_1000@x20 epigenomics_1000@x100 inspiral_100@x1 inspiral_100@x20 inspiral_100@x100 inspiral_400@x1 inspiral_400@x20 inspiral_400@x100 inspiral_1000@x1 inspiral_1000@x20 inspiral_1000@x100 montage_100@x1 montage_100@x20 montage_100@x100 montage_400@x1 montage_400@x20 montage_400@x100 montage_1000@x1 montage_1000@x20 montage_1000@x100 sipht_100@x1 sipht_100@x20 sipht_100@x100 sipht_400@x1 sipht_400@x20 sipht_400@x100 sipht_1000@x1 sipht_1000@x20 sipht_1000@x100 --workers 12 >> results\experiment_suite_b.log 2>&1
echo exit %errorlevel% >> results\experiment_suite_b.log
