from .kernel import ConvKernel, EMAKernel
from .ssm import SSMKernelDense, SSMKernelReal, SSMKernelDiag, SSMKernelDPLR

registry = {
    'conv': ConvKernel,
    'ema': EMAKernel,
    'dense': SSMKernelDense,
    'slow': SSMKernelDense,
    'real': SSMKernelReal,
    's4d': SSMKernelDiag,
    'diag': SSMKernelDiag,
    's4': SSMKernelDPLR,
    'nplr': SSMKernelDPLR,
    'dplr': SSMKernelDPLR,
}
