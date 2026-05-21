# workload

| Model | N | Test@best-eval B-Acc | Paper B-Acc | Delta | Final-test B-Acc | Best-test B-Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| biot | 5 | 65.98 ± 2.22 | 63.98 | 2.00 | 65.84 ± 2.25 | 70.54 ± 1.33 |
| cbramod | 4 | 55.98 ± 5.08 | 71.94 | -15.96 | 51.60 ± 0.67 | 58.05 ± 2.48 |
| labram | 5 | 55.46 ± 2.10 | 55.82 | -0.36 | 57.64 ± 2.67 | 59.62 ± 1.00 |

| Model | Seed | Status | Best eval epoch | Eval B-Acc | Test@best-eval | Final-test | Best-test | Log |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| biot | 42 | ok | 11 | 53.80 | 64.90 | 65.80 | 70.30 | results/eegfm_bench/slurm_logs/efm_biot_workload_t2_s42_14013245.out |
| biot | 43 | ok | 41 | 67.80 | 65.30 | 64.50 | 72.80 | results/eegfm_bench/slurm_logs/efm_biot_workload_t2_s43_14013246.out |
| biot | 44 | ok | 15 | 57.30 | 65.80 | 62.90 | 69.40 | results/eegfm_bench/slurm_logs/efm_biot_workload_t2_s44_14013248.out |
| biot | 45 | ok | 14 | 60.20 | 64.10 | 68.50 | 70.40 | results/eegfm_bench/slurm_logs/efm_biot_workload_t2_s45_14013249.out |
| biot | 46 | ok | 35 | 63.10 | 69.80 | 67.50 | 69.80 | results/eegfm_bench/slurm_logs/efm_biot_workload_t2_s46_14013250.out |
| cbramod | 42 | no_metrics |  |  |  |  |  | results/eegfm_bench/slurm_logs/efm_cbramod_workload_t2_s42_14013251.out |
| cbramod | 43 | ok | 16 | 63.80 | 49.40 | 51.40 | 56.90 | results/eegfm_bench/slurm_logs/efm_cbramod_workload_t2_s43_14013252.out |
| cbramod | 44 | ok | 9 | 65.30 | 59.00 | 51.80 | 59.00 | results/eegfm_bench/slurm_logs/efm_cbramod_workload_t2_s44_14013253.out |
| cbramod | 45 | ok | 8 | 65.10 | 54.70 | 52.40 | 55.30 | results/eegfm_bench/slurm_logs/efm_cbramod_workload_t2_s45_14013254.out |
| cbramod | 46 | ok | 9 | 65.30 | 60.80 | 50.80 | 61.00 | results/eegfm_bench/slurm_logs/efm_cbramod_workload_t2_s46_14013255.out |
| labram | 42 | ok | 25 | 60.00 | 56.70 | 53.50 | 59.00 | results/eegfm_bench/slurm_logs/efm_labram_workload_t2_s42_14013256.out |
| labram | 43 | ok | 10 | 62.40 | 55.30 | 56.60 | 58.80 | results/eegfm_bench/slurm_logs/efm_labram_workload_t2_s43_14013258.out |
| labram | 44 | ok | 27 | 60.00 | 52.20 | 60.30 | 60.50 | results/eegfm_bench/slurm_logs/efm_labram_workload_t2_s44_14013259.out |
| labram | 45 | ok | 15 | 61.30 | 55.30 | 58.90 | 58.90 | results/eegfm_bench/slurm_logs/efm_labram_workload_t2_s45_14013260.out |
| labram | 46 | ok | 14 | 59.60 | 57.80 | 58.90 | 60.90 | results/eegfm_bench/slurm_logs/efm_labram_workload_t2_s46_14013261.out |
