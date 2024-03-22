import math
import json
import argparse
import itertools
from pathlib import Path

from tuner import KernelTuner


def forward_params_list(N):
    blocksize_params = ('MAX_BLOCK_SIZE_VALUE', [64, 128, 256, 512, 1024])
    thread_value_default = [2, 4, 8, 16, 32, 32, 32, 32, 32, 32]
    thread_values_supported = [2, 4, 8, 16, 32, 64, 128]
    log_N_half = int(math.log2(N)) - 1
    thread_values = []
    for val in thread_values_supported:
        if val <= N // 2:
            array = list(thread_value_default)
            array[log_N_half - 1] = val
            thread_values.append('{' + ', '.join(str(v) for v in array) + '}')
    thread_params = ('ITEMS_PER_THREAD_SYM_FWD_VALUES', thread_values)
    value_prod = itertools.product(thread_params[1], blocksize_params[1])
    params_list = [{thread_params[0]: value[0], blocksize_params[0]: value[1]}
                   for value in value_prod]
    return params_list


def backward_params_list(L):
    thread_value_supported = [8, 16, 32, 64, 128]
    thread_params = ('ITEMS_PER_THREAD_SYM_BWD_VALUE', [v for v in thread_value_supported
                                                        if (L + v - 1) // v <= 1024])
    params_list = [{thread_params[0]: value} for value in thread_params[1]]
    return params_list


parser = argparse.ArgumentParser(description='Tuning Cauchy multiply')
parser.add_argument('--mode', default='forward', choices=['forward', 'backward'])
parser.add_argument('-N', default=64, type=int)
parser.add_argument('-L', default=2 ** 14, type=int)
parser.add_argument('--filename', default='tuning_result.json')


if __name__ == '__main__':
    args = parser.parse_args()

    extension_dir = Path(__file__).absolute().parent
    source_files = ['cauchy_cuda.cu']
    if args.mode == 'forward':
        params_list = forward_params_list(args.N)
        tuner = KernelTuner(extension_dir, source_files, params_list,
                            benchmark_script='benchmark_cauchy_tune.py',
                            benchmark_args=['--mode', 'forward', '-N', str(args.N), '-L', '16384'],
                            npool=16)
    else:
        params_list = backward_params_list(args.L)
        tuner = KernelTuner(extension_dir, source_files, params_list,
                            benchmark_script='benchmark_cauchy_tune.py',
                            benchmark_args=['--mode', 'backward', '-N', '64', '-L', str(args.L)],
                            npool=16)

    result = tuner.tune()
    with open(args.filename, 'w') as f:
        json.dump(result, f)
