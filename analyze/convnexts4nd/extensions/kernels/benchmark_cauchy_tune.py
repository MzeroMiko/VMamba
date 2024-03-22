import importlib
import json
import argparse

import torch

from benchmark.utils import benchmark_forward


def generate_data(batch_size, N, L, symmetric=True, device='cuda'):
    if not symmetric:
        v = torch.randn(batch_size, N, dtype=torch.complex64, device=device, requires_grad=True)
        w = torch.randn(batch_size, N, dtype=torch.complex64, device=device, requires_grad=True)
        z = torch.randn(L, dtype=torch.complex64, device=device)
    else:
        assert N % 2 == 0
        v_half = torch.randn(batch_size, N // 2, dtype=torch.complex64, device=device)
        v = torch.cat([v_half, v_half.conj()], dim=-1).requires_grad_(True)
        w_half = torch.randn(batch_size, N // 2, dtype=torch.complex64, device=device)
        w = torch.cat([w_half, w_half.conj()], dim=-1).requires_grad_(True)
        z = torch.exp(1j * torch.randn(L, dtype=torch.float32, device=device))
    return v, z, w


parser = argparse.ArgumentParser(description='Tuning Cauchy multiply')
parser.add_argument('--name', default='cauchy_mult')
parser.add_argument('--mode', default='forward', choices=['forward', 'backward'])
parser.add_argument('-bs', '--batch-size', default=1024, type=int)
parser.add_argument('-N', default=64, type=int)
parser.add_argument('-L', default=2 ** 14, type=int)


if __name__ == '__main__':
    args = parser.parse_args()
    device = 'cuda'
    bs = args.batch_size
    N = args.N
    L = args.L
    repeat = 30
    v, z, w = generate_data(bs, N, L, symmetric=True)
    v_half = v[:, :N // 2].clone().detach().requires_grad_(True)
    w_half = w[:, :N // 2].clone().detach().requires_grad_(True)

    tuning_extension_name = args.name
    # print('Extension name:', tuning_extension_name)
    module = importlib.import_module(tuning_extension_name)
    if args.mode == 'forward':
        _, m = benchmark_forward(repeat, module.cauchy_mult_sym_fwd, v_half, z, w_half,
                                 verbose=False, desc='Cauchy mult symmetric fwd')
    else:
        out = module.cauchy_mult_sym_fwd(v_half, z, w_half)
        dout = torch.randn_like(out)
        _, m = benchmark_forward(repeat, module.cauchy_mult_sym_bwd, v_half, z, w_half, dout,
                                 verbose=False, desc='Cauchy mult symmetric bwd')
    result_dict = dict(time_mean = m.mean, time_iqr = m.iqr)
    print(json.dumps(result_dict))
