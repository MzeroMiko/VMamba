"""Implementation of UnICORNN model.

Adapted from https://github.com/tk-rusch/unicornn/blob/main/health_care/network.py.

Original docstring:
This code implements a fast CUDA version of the stacked UnICORNN model.
We emphasise that this code builds up on the fast CUDA implementation of the IndRNN https://github.com/Sunnydreamrain/IndRNN_pytorch.
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Parameter
from collections import namedtuple

from src.models.sequence.base import SequenceModule, TransposedModule

try:
    from cupy.cuda import function
    from pynvrtc.compiler import Program

    _unicornn_available = True
except ImportError:
    _unicornn_available = False


UnICORNN_CODE = """
extern "C" {
    __forceinline__ __device__ float sigmoid(float x)
    {
        return (float) 1./(1.+exp(-x));
    }
    __forceinline__ __device__ float sigmoid_grad(float x)
    {
        return (float) exp(-x)/((1.+exp(-x))*(1.+exp(-x)));
    }
    __forceinline__ __device__ float activation(float x)
    {
        return (float)tanh(x);
    }
    __forceinline__ __device__ float calc_grad_activation(float x)
    {
        return (float)1/(cosh(x)*cosh(x));
    }
    __global__ void unicornn_fwd( const float * __restrict__ x,
                            const float * __restrict__ weight_hh, const float * __restrict__ hy_initial,
                            const float * __restrict__ hz_initial, float * __restrict__ hy_final,
                            float * __restrict__ hz_final,
                            const int len, const int batch, const int d_model, const float * __restrict__ c,
                            double dt, double alpha,
                            float * __restrict__ hy_all)
    {
        int ncols = batch*d_model;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        const float weight_hh_cur = *(weight_hh + (col%d_model));
        const float c_cur = *(c + (col%d_model));
        float hy = *(hy_initial + col);
        float hz = *(hz_initial + col);
        const float *xp = x+col;
        float *hy_all_p = hy_all+col;
        for (int row = 0; row < len; ++row)
        {
            hz -= dt*sigmoid(c_cur)*(activation(hy*weight_hh_cur+(*xp))+alpha*hy);
            hy += dt*sigmoid(c_cur)*hz;
            *hy_all_p = hy;
            xp += ncols;
            hy_all_p += ncols;
        }
        *(hy_final + col) = hy;
        *(hz_final + col) = hz;
    }
    __global__ void unicornn_bwd(const float * __restrict__ x,
                             const float * __restrict__ weight_hh, const float * __restrict__ hy_final,
                             const float * __restrict__ hz_final,
                            const float * __restrict__ grad_h,
                            const int len, const int batch, const int d_model, const float * __restrict__ c,
                            double dt, double alpha, float * __restrict__ grad_x,
                            float * __restrict__ grad_weight_hh, float * __restrict__ grad_c)
    {
        int ncols = batch*d_model;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        const float weight_hh_cur = *(weight_hh + (col%d_model));
        const float c_cur = *(c + (col%d_model));
        float gweight_hh = 0;
        float gc = 0;
        const float *xp = x+col + (len-1)*ncols;
        float *gxp = grad_x + col + (len-1)*ncols;
        const float *ghp = grad_h + col + (len-1)*ncols;
        float delta_z = 0;
        float delta_y = (*ghp);
        float delta_dt = 0;
        float hy = *(hy_final + col);
        float hz = *(hz_final + col);
        for (int row = len-1; row >= 0; --row)
        {
            delta_dt = delta_y*dt*sigmoid_grad(c_cur)*hz;
            // reconstruct hidden states based on the final hidden state using adjoint symplectic Euler:
            hy=hy-dt*sigmoid(c_cur)*hz;
            hz=hz+dt*sigmoid(c_cur)*(activation(hy*weight_hh_cur+(*xp))+alpha*hy);
            delta_z += delta_y*dt*sigmoid(c_cur);
            gweight_hh -= delta_z*dt*sigmoid(c_cur)*calc_grad_activation(hy*weight_hh_cur+(*xp))*hy;
            gc += delta_dt-delta_z*(dt*sigmoid_grad(c_cur)*(activation(hy*weight_hh_cur+(*xp))+alpha*hy));
            *gxp = -delta_z*dt*sigmoid(c_cur)*calc_grad_activation(hy*weight_hh_cur+(*xp));
            if(row==0)break;
            ghp -= ncols;
            delta_y += -delta_z*dt*sigmoid(c_cur)*(calc_grad_activation(hy*weight_hh_cur+(*xp))*weight_hh_cur+alpha) + (*ghp);
            xp -= ncols;
            gxp -= ncols;
        }
        atomicAdd(grad_weight_hh + (col%d_model), gweight_hh);
        atomicAdd(grad_c + (col%d_model), gc);
    }
}
"""


class UnICORNN_compile:
    if _unicornn_available:
        _UnICORNN_PROG = Program(UnICORNN_CODE, "unicornn_prog.cu")
        _UnICORNN_PTX = _UnICORNN_PROG.compile()
        _DEVICE2FUNC = {}

    def __init__(self):
        super(UnICORNN_compile, self).__init__()

    def compile_functions(self):
        device = torch.cuda.current_device()
        mod = function.Module()
        mod.load(bytes(self._UnICORNN_PTX.encode()))
        fwd_func = mod.get_function("unicornn_fwd")
        bwd_func = mod.get_function("unicornn_bwd")

        Stream = namedtuple("Stream", ["ptr"])
        current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)

        self._DEVICE2FUNC[device] = (current_stream, fwd_func, bwd_func)
        return current_stream, fwd_func, bwd_func

    def get_functions(self):
        res = self._DEVICE2FUNC.get(torch.cuda.current_device(), None)
        return res if res else self.compile_functions()


class UnICORNN_Compute_GPU(Function):
    @staticmethod
    def forward(ctx, x, weight_hh, hy_initial, hz_initial, c, alpha, dt):
        comp = UnICORNN_compile()
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d_model = x.size(-1)
        ncols = batch * d_model
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) // thread_per_block + 1

        size = (length, batch, d_model) if x.dim() == 3 else (batch, d_model)
        hy_all = x.new(*size)

        hy_final = x.new(batch, d_model)
        hz_final = x.new(batch, d_model)

        stream, fwd_func, _ = comp.get_functions()
        FUNC = fwd_func
        FUNC(
            args=[
                x.contiguous().data_ptr(),
                weight_hh.contiguous().data_ptr(),
                hy_initial.contiguous().data_ptr(),
                hz_initial.contiguous().data_ptr(),
                hy_final.contiguous().data_ptr(),
                hz_final.contiguous().data_ptr(),
                length,
                batch,
                d_model,
                c.contiguous().data_ptr(),
                dt.item(),
                alpha.item(),
                hy_all.contiguous().data_ptr(),
            ],
            block=(thread_per_block, 1, 1),
            grid=(num_block, 1, 1),
            stream=stream,
        )

        ctx.save_for_backward(x, weight_hh, hy_final, hz_final, c, alpha, dt)
        return hy_all

    @staticmethod
    def backward(ctx, grad_h):
        x, weight_hh, hy_final, hz_final, c, alpha, dt = ctx.saved_tensors
        comp = UnICORNN_compile()
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d_model = x.size(-1)
        ncols = batch * d_model
        thread_per_block = min(256, ncols)
        num_block = (ncols - 1) // thread_per_block + 1

        grad_x = x.new(*x.size())
        grad_weight_hh = x.new(d_model).zero_()
        grad_c = x.new(d_model).zero_()

        stream, _, bwd_func = comp.get_functions()
        FUNC = bwd_func
        FUNC(
            args=[
                x.contiguous().data_ptr(),
                weight_hh.contiguous().data_ptr(),
                hy_final.contiguous().data_ptr(),
                hz_final.contiguous().data_ptr(),
                grad_h.contiguous().data_ptr(),
                length,
                batch,
                d_model,
                c.contiguous().data_ptr(),
                dt.item(),
                alpha.item(),
                grad_x.contiguous().data_ptr(),
                grad_weight_hh.contiguous().data_ptr(),
                grad_c.contiguous().data_ptr(),
            ],
            block=(thread_per_block, 1, 1),
            grid=(num_block, 1, 1),
            stream=stream,
        )

        return grad_x, grad_weight_hh, None, None, grad_c, None, None


class UnICORNN_recurrence(nn.Module):
    def __init__(self, d_model, dt, alpha):
        super(UnICORNN_recurrence, self).__init__()
        self.d_model = d_model
        self.dt = torch.tensor(dt)
        self.c_ = Parameter(torch.Tensor(d_model))
        self.alpha = torch.tensor(alpha)
        self.weight_hh = Parameter(torch.Tensor(d_model))
        self.reset_parameters()

    def reset_parameters(self):
        for name, weight in self.named_parameters():
            if "weight_hh" in name:
                nn.init.uniform_(weight, a=0, b=1)
            if "c_" in name:
                nn.init.uniform_(weight, a=-0.1, b=0.1)

    def forward(self, input):
        hy0, hz0 = (
            input.data.new(input.size(-2), input.size(-1)).zero_(),
            input.data.new(input.size(-2), input.size(-1)).zero_(),
        )
        return UnICORNN_Compute_GPU.apply(
            input, self.weight_hh, hy0, hz0, self.c_, self.alpha, self.dt
        )


class Dropout_overtime(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p=0.5, training=False):
        output = input.clone()
        noise = input.data.new(input.size(-2), input.size(-1))
        if training:
            noise.bernoulli_(1 - p).div_(1 - p)
            noise = noise.unsqueeze(0).expand_as(input)
            output.mul_(noise)
        ctx.save_for_backward(noise)
        ctx.training = training
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (noise,) = ctx.saved_tensors
        if ctx.training:
            return grad_output.mul(noise), None, None
        else:
            return grad_output, None, None


dropout_overtime = Dropout_overtime.apply


class LinearInitovertime(nn.Module):
    def __init__(self, d_input, d_model, bias=True):
        super(LinearInitovertime, self).__init__()
        self.fc = nn.Linear(d_input, d_model, bias=bias)
        self.d_input = d_input
        self.d_model = d_model

    def forward(self, x):
        y = x.contiguous().view(-1, self.d_input)
        y = self.fc(y)
        y = y.view(x.size()[0], x.size()[1], self.d_model)
        return y


@TransposedModule
class UnICORNN(SequenceModule):
    def __init__(
        self,
        # d_input,
        # d_output,
        # l_output,
        d_model,
        dt,
        alpha,
        n_layers,
        dropout=0.1,
        **kwargs
    ):
        if not _unicornn_available:
            raise ImportError(
                "Check unicornn codebase for install instructions. Requires cupy and pynvrtc."
            )

        super(UnICORNN, self).__init__()
        self.d_model = d_model
        self.d_output = d_model
        self.dropout = dropout
        self.nlayers = n_layers
        # self.l_output = l_output
        self.DIs = nn.ModuleList()
        # denseinput = LinearInitovertime(d_input, nhid)
        # self.DIs.append(denseinput)
        # for x in range(self.nlayers - 1):
        for x in range(self.nlayers):
            denseinput = LinearInitovertime(d_model, d_model)
            self.DIs.append(denseinput)
        # self.classifier = nn.Linear(nhid, d_output)
        self.init_weights()
        self.RNNs = []
        for x in range(self.nlayers):
            rnn = UnICORNN_recurrence(d_model, dt[x], alpha)
            self.RNNs.append(rnn)
        self.RNNs = torch.nn.ModuleList(self.RNNs)

    def init_weights(self):
        for name, param in self.named_parameters():
            if ("fc" in name) and "weight" in name:
                nn.init.kaiming_uniform_(param, a=8, mode="fan_in")
            # if "classifier" in name and "weight" in name:
            #     nn.init.kaiming_normal_(param.data)
            if "bias" in name:
                param.data.fill_(0.0)

    def forward(self, input, *args, **kwargs):

        input = input.transpose(0, 1)

        rnnoutputs = {}
        rnnoutputs["outlayer-1"] = input
        for x in range(len(self.RNNs)):
            rnnoutputs["dilayer%d" % x] = self.DIs[x](
                rnnoutputs["outlayer%d" % (x - 1)]
            )
            rnnoutputs["outlayer%d" % x] = self.RNNs[x](rnnoutputs["dilayer%d" % x])
            rnnoutputs["outlayer%d" % x] = dropout_overtime(
                rnnoutputs["outlayer%d" % x], self.dropout, self.training
            )

        # temp = rnnoutputs["outlayer%d" % (len(self.RNNs) - 1)][-1]
        # output = self.classifier(temp)
        output = rnnoutputs["outlayer%d" % (len(self.RNNs) - 1)]
        output = output.transpose(0, 1)

        # if self.l_output == 0:
        #     output = output[:, -1]
        # else:
        #     output = output[:, -self.l_output :]

        return output
