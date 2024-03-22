import os
import shutil
import subprocess
import sys
# import tempfile
# import importlib
import random
import string
import json


from functools import partial
from multiprocessing import Pipe, Pool, Process
from pathlib import Path

from tqdm import tqdm

import numpy as np


def read_file(filename):
    """ return the contents of the file named filename or None if file not found """
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            return f.read()


def write_file(filename, string):
    """dump the contents of string to a file called filename"""
    with open(filename, 'w', encoding="utf-8") as f:
        f.write(string)


def prepare_kernel_string(kernel_string, params):
    for k, v in params.items():
        kernel_string = "#define " + k + " " + str(v) + "\n" + kernel_string
    return kernel_string


def compile_extension(temp_dir, install=False, verbose=True):
    # Need to copy this process's environments, otherwise it can't find the compilers
    env = {**os.environ,
           'TUNING_SOURCE_DIR': str(temp_dir),
           'TUNING_EXTENSION_NAME': str(temp_dir.stem)}
    # https://stackoverflow.com/questions/53173314/how-to-change-distutils-output-directory
    # Need separate build directories for parallel compilation
    output = subprocess.run(
        # [sys.executable, "tuning_setup.py", 'build', f'--build-base={str(temp_dir)}',
        #  f'--build-lib={str(temp_dir)}'],
        [sys.executable, "tuning_setup.py", 'build' if not install else 'develop'],
        cwd=temp_dir,
        env=env,
        capture_output=True,
        # check=True
    )
    if verbose:
        print(output)
        print('Done compiling' if not install else 'Done installing')


def uninstall_extensions(tuning_extension_names, verbose=True):
    # Need to copy this process's environments, otherwise it can't find the compilers
    env = {**os.environ}
    output = subprocess.run(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', *tuning_extension_names],
        env=env,
        capture_output=True,
        # check=True
    )
    if verbose:
        print(output)
        print('Done uninstalling')


def benchmark_extension(benchmark_script, *benchmark_args, verbose=True):
    # Need to copy this process's environments, otherwise it can't find the compilers
    env = os.environ
    # https://stackoverflow.com/questions/53173314/how-to-change-distutils-output-directory
    # Need separate build directories for parallel compilation
    process = subprocess.run(
        [sys.executable, benchmark_script, *benchmark_args],
        env=os.environ,
        capture_output=True,
        # check=True
    )
    if verbose:
        print(process)
        print('Done benchmarking')
    return json.loads(process.stdout.decode(sys.stdout.encoding))


# def benchmark(connection, temp_dir):
#     import torch
#     # module = importlib.import_module(tuning_extension_name)
#     torch.ops.load_library(temp_dir / 'torch_butterfly_tuning.so')
#     batch_size = 1024
#     n = 32
#     twiddle = torch.randn(1, 1, 5, n // 2, 2, 2, device='cuda')
#     input = torch.randn(batch_size, 1, n, device=twiddle.device)
#     output = torch.ops.torch_butterfly.butterfly_multiply_fw(twiddle, input, True)
#     # https://medium.com/@auro_227/timing-your-pytorch-code-fragments-e1a556e81f2
#     res = []
#     for _ in range(32):
#         start = torch.cuda.Event(enable_timing=True)
#         end = torch.cuda.Event(enable_timing=True)
#         start.record()
#         output = torch.ops.torch_butterfly.butterfly_multiply_fw(twiddle, input, True)
#         end.record()
#         torch.cuda.synchronize()
#         res.append(start.elapsed_time(end))
#     print(output.shape)
#     res = np.array(res)
#     connection.send((np.mean(res), np.std(res)))


def set_up_tuning_temp_dir(params: dict, source_files, extension_dir, verbose=True):
    if verbose:
        print('params: ', params)
    # TD [2021-10-22]: tempfile.mkdtemp sometimes create dir name with '_' in it, thus messing up
    # the extension name.
    # temp_dir = Path(tempfile.mkdtemp(prefix="temp_", dir=Path.cwd().parent)).absolute()
    tuning_extension_name = 'temp_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    temp_dir = (Path.cwd().parent / tuning_extension_name).absolute()
    if temp_dir.exists():
        shutil.rmtree(temp_dir)  # shutil.copytree doesn't want directory that already exists
    shutil.copytree(extension_dir, temp_dir)
    sources = [temp_dir / name for name in source_files]
    for kernel_source in sources:
        ks = read_file(kernel_source)
        ks = prepare_kernel_string(ks, params)
        write_file(kernel_source, ks)
    return temp_dir


class KernelTuner:

    def __init__(self, extension_dir, source_files, params_list, benchmark_script,
                 benchmark_args, npool=8, verbose=True):
        self.extension_dir = extension_dir
        self.source_files = source_files
        self.params_list = params_list
        self.benchmark_script = benchmark_script
        self.benchmark_args = benchmark_args
        self.npool = npool
        self.verbose = verbose

    def tune(self):
        temp_dirs = [set_up_tuning_temp_dir(params, self.source_files, self.extension_dir,
                                            verbose=self.verbose)
                     for params in self.params_list]
        # Compile in parallel (for speed), then install sequentially to ensure correctness
        with Pool(self.npool) as p:
            p.map(compile_extension, temp_dirs)
        # with Pool(1) as p:
        #     p.map(partial(compile_extension, install=True), [temp_dirs])
        for temp_dir in tqdm(temp_dirs):
            try:
                compile_extension(temp_dir, install=True)
            except:
                pass
        # # We benchmark on a separate process so that they can import the extension that just got compiled.
        # for params, temp_dir in params_tempdir:
        #     print('Benchmarking: ', params)
        #     recv_conn, send_conn = Pipe(duplex=False)
        #     benchmark_process = Process(target=benchmark_fwd, args=(send_conn, str(temp_dir.stem)))
        #     benchmark_process.start()
        #     result = recv_conn.recv()
        #     benchmark_process.join()
        #     print('result', result)
        results = []
        for params, temp_dir in tqdm(list(zip(self.params_list, temp_dirs))):
            try:
                results.append((params,
                                benchmark_extension(self.benchmark_script,
                                                    *['--name', temp_dir.stem] + self.benchmark_args)))
            except:
                pass
        print(results)
        uninstall_extensions([temp_dir.stem for temp_dir in temp_dirs])
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir)
        return results
