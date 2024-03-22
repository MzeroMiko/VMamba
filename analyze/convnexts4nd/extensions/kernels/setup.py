from setuptools import setup
import torch.cuda
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME

ext_modules = []
if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        'structured_kernels', [
            'cauchy.cpp',
            'cauchy_cuda.cu',
        ],
        extra_compile_args={'cxx': ['-g', '-march=native', '-funroll-loops'],
                            # 'nvcc': ['-O2', '-lineinfo']
                            'nvcc': ['-O2', '-lineinfo', '--use_fast_math']
                            }
    )
    ext_modules.append(extension)

setup(
    name='structured_kernels',
    version="0.1.0",
    ext_modules=ext_modules,
    # cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)})
    cmdclass={'build_ext': BuildExtension})
