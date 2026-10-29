@echo off
rem Short test of the L7 learner fix (block decisions every 50 generations + HV_jin reward) on the tch backbone.
rem 2 training seeds x 24 episodes, 4 test cases x 4 seeds; baselines reused from results\experiments\suite_a_tch.
rem Launch detached:  powershell -Command "Start-Process cmd -ArgumentList '/c scripts\run_dqn_fix_test.cmd' -WindowStyle Hidden"
cd /d %~dp0..
.venv\Scripts\python.exe -u scripts\experiment.py --name dqn_fix_test --reuse-from suite_a_tch --policies tch_trained_blk tch_trained_nogat_blk tch_fixed_fav tch_conv imoead tch_trained_stable tch_trained_nogat_stable --reference tch_fixed_fav --rank-metric IGD --energy-scope device --external-io device --makespan sum_apps --train-episodes 24 --generations 1000 --train-generations 1000 --seeds 4 --train-seeds 0 1 --train-set small_20_i1@x1 small_20_i1@x20 small_50_i1@x1 small_50_i1@x20 basic_60_i1@x1 basic_60_i1@x20 basic_100_i1@x1 basic_100_i1@x20 medium_100_i1@x1 medium_100_i1@x20 medium_250_i1@x1 medium_250_i1@x20 --test-set small_20_i0@x1 basic_60_i0@x20 medium_100_i0@x1 medium_250_i0@x20 --workers 12 >> results\experiment_dqn_fix_test.log 2>&1
echo exit %errorlevel% >> results\experiment_dqn_fix_test.log
