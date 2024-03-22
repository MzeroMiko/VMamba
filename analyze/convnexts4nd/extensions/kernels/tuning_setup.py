import os
from setuptools import setup
from pathlib import Path

import torch.cuda
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME


extensions_dir = Path(os.getenv('TUNING_SOURCE_DIR')).absolute()
assert extensions_dir.exists()
source_files=[
    'cauchy.cpp',
    'cauchy_cuda.cu',
]
sources = [str(extensions_dir / name) for name in source_files]

extension_name = os.getenv('TUNING_EXTENSION_NAME', default='cauchy_mult_tuning')
ext_modules = []
if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        extension_name,
        sources,
        include_dirs=[extensions_dir],
        extra_compile_args={'cxx': ['-g', '-march=native', '-funroll-loops'],
                            # 'nvcc': ['-O2', '-lineinfo']
                            'nvcc': ['-O2', '-lineinfo', '--use_fast_math']
                            }
    )
    ext_modules.append(extension)

setup(
    name=extension_name,
    ext_modules=ext_modules,
    # cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)})
    cmdclass={'build_ext': BuildExtension})
