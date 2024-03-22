#!/usr/bin/env bash
python tune_cauchy.py --mode forward -N 4 --filename tuning_result_fwd_N_4.json
python tune_cauchy.py --mode forward -N 8 --filename tuning_result_fwd_N_8.json
python tune_cauchy.py --mode forward -N 16 --filename tuning_result_fwd_N_16.json
python tune_cauchy.py --mode forward -N 32 --filename tuning_result_fwd_N_32.json
python tune_cauchy.py --mode forward -N 64 --filename tuning_result_fwd_N_64.json
python tune_cauchy.py --mode forward -N 128 --filename tuning_result_fwd_N_128.json
python tune_cauchy.py --mode forward -N 256 --filename tuning_result_fwd_N_256.json
python tune_cauchy.py --mode forward -N 512 --filename tuning_result_fwd_N_512.json

python tune_cauchy.py --mode backward -L 1024 --filename tuning_result_bwd_L_1k.json
python tune_cauchy.py --mode backward -L 2048 --filename tuning_result_bwd_L_2k.json
python tune_cauchy.py --mode backward -L 4096 --filename tuning_result_bwd_L_4k.json
python tune_cauchy.py --mode backward -L 8192 --filename tuning_result_bwd_L_8k.json
python tune_cauchy.py --mode backward -L 16384 --filename tuning_result_bwd_L_16k.json
