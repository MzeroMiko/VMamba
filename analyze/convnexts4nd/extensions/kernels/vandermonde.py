import torch

from structured_kernels import vand_log_mult_sym_fwd, vand_log_mult_sym_bwd

def log_vandermonde_cuda(v, z, L):
    """ Wrap the cuda method to deal with shapes """
    v, z = torch.broadcast_tensors(v, z)
    shape = v.shape

    v = v.contiguous()
    z = z.contiguous()

    N = v.size(-1)
    assert z.size(-1) == N
    y = LogVandMultiplySymmetric.apply(v.view(-1, N), z.view(-1, N), L)
    y = y.view(*shape[:-1], L)
    return y

class LogVandMultiplySymmetric(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v, x, L):
        batch, N = v.shape
        supported_N_values = [1 << log_n for log_n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        if not N in supported_N_values:
            raise NotImplementedError(f'Only support N values in {supported_N_values}')
        max_L_value = 32 * 1024 * 64 * 1024
        if L > max_L_value:
            raise NotImplementedError(f'Only support L values <= {max_L_value}')
        if not v.is_cuda and x.is_cuda:
            raise NotImplementedError(f'Only support CUDA tensors')
        ctx.save_for_backward(v, x)
        return vand_log_mult_sym_fwd(v, x, L)

    @staticmethod
    def backward(ctx, dout):
        v, x = ctx.saved_tensors
        dv, dx = vand_log_mult_sym_bwd(v, x, dout)
        return dv, dx, None


if vand_log_mult_sym_fwd and vand_log_mult_sym_bwd is not None:
    log_vandermonde_fast = LogVandMultiplySymmetric.apply
else:
    log_vandermonde_fast = None
