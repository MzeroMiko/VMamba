"""Implementation of Continuous Kernel Convolution (CKConv).
Paper: https://arxiv.org/abs/2102.02611

Adapted directly from https://github.com/dwromero/ckconv.
"""

from typing import Tuple, Optional

import numpy as np
import torch
import torch.fft
import torch.fft
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils import weight_norm


def Linear1d(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        bias: bool = True,
) -> torch.nn.Module:
    """
    Implements a Linear Layer in terms of a point-wise convolution.
    """
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def Linear2d(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        bias: bool = True,
) -> torch.nn.Module:
    """
    Implements a Linear Layer in terms of a point-wise convolution.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def Swish():
    """
    out = x * sigmoid(x)
    """
    return Expression(lambda x: x * torch.sigmoid(x))


def Sine():
    """
    out = sin(x)
    """
    return Expression(lambda x: torch.sin(x))


class LayerNorm(nn.Module):
    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-12,
    ):
        """Uses GroupNorm implementation with group=1 for speed."""
        super().__init__()
        # we use GroupNorm to implement this efficiently and fast.
        self.layer_norm = torch.nn.GroupNorm(1, num_channels=num_channels, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)


# From LieConv
class Expression(torch.nn.Module):
    def __init__(self, func):
        """
        Creates a torch.nn.Module that applies the function func.
        :param func: lambda function
        """
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def Multiply(
        omega_0: float,
):
    """
    out = omega_0 * x
    """
    return Expression(lambda x: omega_0 * x)


def causal_padding(
        x: torch.Tensor,
        kernel: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 1. Pad the input signal & kernel tensors.
    # Check if sizes are odd. If not, add a pad of zero to make them odd.
    if kernel.shape[-1] % 2 == 0:
        kernel = f.pad(kernel, [1, 0], value=0.0)
        # x = torch.nn.functional.pad(x, [1, 0], value=0.0)
    # 2. Perform padding on the input so that output equals input in length
    x = f.pad(x, [kernel.shape[-1] - 1, 0], value=0.0)

    return x, kernel


def causal_conv(
        x: torch.Tensor,
        kernel: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Args:
        x: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.
    Returns:
        (Tensor) Convolved tensor
    """

    x, kernel = causal_padding(x, kernel)
    return torch.nn.functional.conv1d(x, kernel, bias=bias, padding=0)


def causal_fftconv(
        x: torch.Tensor,
        kernel: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        double_precision: bool = False,
) -> torch.Tensor:
    """
    Args:
        x: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.
    Returns:
        (Tensor) Convolved tensor
    """

    x_shape = x.shape
    # 1. Handle padding of the input and the kernel to make them odd.
    x, kernel = causal_padding(x, kernel)

    # 2. Pad the kernel tensor to make them equally big. Required for fft.
    kernel = f.pad(kernel, [0, x.size(-1) - kernel.size(-1)])

    # 3. Perform fourier transform
    if double_precision:
        # We can make usage of double precision to make more accurate approximations of the convolution response.
        x = x.double()
        kernel = kernel.double()

    x_fr = torch.fft.rfft(x, dim=-1)
    kernel_fr = torch.fft.rfft(kernel, dim=-1)

    # 4. Multiply the transformed matrices:
    # (Input * Conj(Kernel)) = Correlation(Input, Kernel)
    kernel_fr = torch.conj(kernel_fr)
    output_fr = (x_fr.unsqueeze(1) * kernel_fr.unsqueeze(0)).sum(
        2
    )  # 'ab..., cb... -> ac...'

    # 5. Compute inverse FFT, and remove extra padded values
    # Once we are back in the spatial domain, we can go back to float precision, if double used.
    out = torch.fft.irfft(output_fr, dim=-1).float()

    out = out[:, :, : x_shape[-1]]

    # 6. Optionally, add a bias term before returning.
    if bias is not None:
        out = out + bias.view(1, -1, 1)

    return out


class KernelNet(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            activation_function: str,
            norm_type: str,
            dim_linear: int,
            bias: bool,
            omega_0: float,
            weight_dropout: float,
    ):
        """
        Creates an 3-layer MLP, which parameterizes a convolutional kernel as:
        relative positions -> hidden_channels -> hidden_channels -> in_channels * out_channels
        :param in_channels:  Dimensionality of the relative positions (Default: 1).
        :param out_channels:  input channels * output channels of the resulting convolutional kernel.
        :param hidden_channels: Number of hidden units.
        :param activation_function: Activation function used.
        :param norm_type: Normalization type used.
        :param dim_linear:  Spatial dimension of the input, e.g., for audio = 1, images = 2 (only 1 suported).
        :param bias:  If True, adds a learnable bias to the layers.
        :param omega_0: Value of the omega_0 value (only used in Sine networks).
        :param weight_dropout: Dropout rate applied to the sampled convolutional kernel.
        """
        super().__init__()

        is_siren = activation_function == "Sine"
        w_dp = weight_dropout != 0.0

        Norm = {
            "BatchNorm": torch.nn.BatchNorm1d,
            "LayerNorm": LayerNorm,
            "": torch.nn.Identity,
        }[norm_type]
        ActivationFunction = {
            "ReLU": torch.nn.ReLU,
            "LeakyReLU": torch.nn.LeakyReLU,
            "Swish": Swish,
            "Sine": Sine,
        }[activation_function]
        Linear = {1: Linear1d, 2: Linear2d}[dim_linear]

        self.kernel_net = torch.nn.Sequential(
            weight_norm(Linear(in_channels, hidden_channels, bias=bias)),
            Multiply(omega_0) if is_siren else torch.nn.Identity(),
            Norm(hidden_channels) if not is_siren else torch.nn.Identity(),
            ActivationFunction(),
            weight_norm(Linear(hidden_channels, hidden_channels, bias=bias)),
            Multiply(omega_0) if is_siren else torch.nn.Identity(),
            Norm(hidden_channels) if not is_siren else torch.nn.Identity(),
            ActivationFunction(),
            weight_norm(Linear(hidden_channels, out_channels, bias=bias)),
            torch.nn.Dropout(p=weight_dropout) if w_dp else torch.nn.Identity(),
        )

        # initialize the kernel function
        self.initialize(
            mean=0.0,
            variance=0.01,
            bias_value=0.0,
            is_siren=(activation_function == "Sine"),
            omega_0=omega_0,
        )

    def forward(self, x):
        return self.kernel_net(x)

    def initialize(self, mean, variance, bias_value, is_siren, omega_0):

        if is_siren:
            # Initialization of SIRENs
            net_layer = 1
            for (i, m) in enumerate(self.modules()):
                if (
                        isinstance(m, torch.nn.Conv1d)
                        or isinstance(m, torch.nn.Conv2d)
                        or isinstance(m, torch.nn.Linear)
                ):
                    if net_layer == 1:
                        m.weight.data.uniform_(
                            -1, 1
                        )  # Normally (-1, 1) / in_dim but we only use 1D inputs.
                        # Important! Bias is not defined in original SIREN implementation!
                        net_layer += 1
                    else:
                        m.weight.data.uniform_(
                            -np.sqrt(6.0 / m.weight.shape[1]) / omega_0,
                            # the in_size is dim 2 in the weights of Linear and Conv layers
                            np.sqrt(6.0 / m.weight.shape[1]) / omega_0,
                        )
                    # Important! Bias is not defined in original SIREN implementation
                    if m.bias is not None:
                        m.bias.data.uniform_(-1.0, 1.0)
        else:
            # Initialization of ReLUs
            net_layer = 1
            intermediate_response = None
            for (i, m) in enumerate(self.modules()):
                if (
                        isinstance(m, torch.nn.Conv1d)
                        or isinstance(m, torch.nn.Conv2d)
                        or isinstance(m, torch.nn.Linear)
                ):
                    m.weight.data.normal_(
                        mean,
                        variance,
                    )
                    if m.bias is not None:

                        if net_layer == 1:
                            # m.bias.data.fill_(bias_value)
                            range = torch.linspace(-1.0, 1.0, steps=m.weight.shape[0])
                            bias = -range * m.weight.data.clone().squeeze()
                            m.bias = torch.nn.Parameter(bias)

                            intermediate_response = [
                                m.weight.data.clone(),
                                m.bias.data.clone(),
                            ]
                            net_layer += 1

                        elif net_layer == 2:
                            range = torch.linspace(-1.0, 1.0, steps=m.weight.shape[0])
                            range = range + (range[1] - range[0])
                            range = (
                                    range * intermediate_response[0].squeeze()
                                    + intermediate_response[1]
                            )

                            bias = -torch.einsum(
                                "oi, i -> o", m.weight.data.clone().squeeze(), range
                            )
                            m.bias = torch.nn.Parameter(bias)

                            net_layer += 1

                        else:
                            m.bias.data.fill_(bias_value)


class CKConv(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            activation_function: str,
            norm_type: str,
            dim_linear: int,
            bias: bool,
            omega_0: float,
            weight_dropout: float,
    ):
        """
        Creates a Continuous Kernel Convolution.
        :param in_channels: Number of channels in the input signal
        :param out_channels: Number of channels produced by the convolution
        :param hidden_channels: Number of hidden units in the network parameterizing the ConvKernel (KernelNet).
        :param activation_function: Activation function used in KernelNet.
        :param norm_type: Normalization type used in KernelNet. (only for non-Sine KernelNets).
        :param dim_linear: patial dimension of the input, e.g., for audio = 1, images = 2 (only 1 suported).
        :param bias: If True, adds a learnable bias to the output.
        :param omega_0: Value of the omega_0 value of the KernelNet. (only for non-Sine KernelNets).
        :param weight_dropout: Dropout rate applied to the sampled convolutional kernels.
        """
        super().__init__()
        self.Kernel = KernelNet(
            dim_linear,
            out_channels * in_channels,
            hidden_channels,
            activation_function,
            norm_type,
            dim_linear,
            bias,
            omega_0,
            weight_dropout,
        )

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.fill_(value=0.0)
        else:
            self.bias = None

        # Non-persistent values
        self.rel_positions = None
        self.sigma = None
        self.sr_change = 1.0

        self.register_buffer("train_length", torch.zeros(1).int(), persistent=True)
        self.register_buffer("conv_kernel", torch.zeros(in_channels), persistent=False)

    def forward(self, x):
        # Construct kernel
        x_shape = x.shape

        rel_pos = self.handle_rel_positions(x)
        conv_kernel = self.Kernel(rel_pos).view(-1, x_shape[1], *x_shape[2:])

        # ---- Different samling rate --------
        # If freq test > freq test, smooth out high-freq elements.
        if self.sigma is not None:
            from math import pi, sqrt, exp

            n = int(1 / self.sr_change) * 2 + 1
            h = n // 2
            G = (
                lambda x: 1
                          / (self.sigma * sqrt(2 * pi))
                          * exp(-float(x) ** 2 / (2 * self.sigma ** 2))
            )

            smoothing_ker = [G(x) for x in range(-h, h + 1)]
            smoothing_ker = torch.Tensor(smoothing_ker).cuda().unsqueeze(0).unsqueeze(0)
            conv_kernel[:, :, h:-h] = torch.conv1d(
                conv_kernel.view(-1, 1, *x_shape[2:]), smoothing_ker, padding=0
            ).view(*conv_kernel.shape[:-1], -1)
        # multiply by the sr_train / sr_test
        if self.sr_change != 1.0:
            conv_kernel *= self.sr_change
        # ------------------------------------

        # For computation of "weight_decay"
        self.conv_kernel = conv_kernel

        # We have noticed that the results of fftconv become very noisy when the length of
        # the input is very small ( < 50 samples). As this might occur when we use subsampling,
        # we replace causal_fftconv by causal_conv in settings where this occurs.
        if x_shape[-1] < self.train_length.item():
            # Use spatial convolution:
            return causal_conv(x, conv_kernel, self.bias)
        else:
            # Otherwise use fft convolution:
            return causal_fftconv(x, conv_kernel, self.bias)

    def handle_rel_positions(self, x):
        """
        Handles the vector or relative positions which is given to KernelNet.
        """
        if self.rel_positions is None:
            if self.train_length[0] == 0:
                # The ckconv has not been trained yet. Set maximum length to be 1.
                self.train_length[0] = x.shape[-1]

            # Calculate the maximum relative position based on the length of the train set,
            # and the current length of the input.
            max_relative_pos = self.calculate_max(
                self.train_length.item(), current_length=x.shape[-1]
            )

            # Creates the vector of relative positions.
            self.rel_positions = (
                torch.linspace(-1.0, max_relative_pos, x.shape[-1])
                    .cuda()
                    .unsqueeze(0)
                    .unsqueeze(0)
            )  # -> With form: [batch_size=1, in_channels=1, x_dimension]

            # calculate and save the sr ratio for later
            if self.train_length.item() > x.shape[-1]:
                self.sr_change = round(self.train_length.item() / x.shape[-1])
            else:
                self.sr_change = 1 / round(x.shape[-1] / self.train_length.item())

            # if new signal has higher frequency
            if self.sr_change < 1:
                self.sigma = 0.5

        return self.rel_positions

    @staticmethod
    def calculate_max(
            train_length: int,
            current_length: int,
    ) -> float:
        """
        Calculates the maximum relative position for the current length based on the input length.
        This is used to avoid kernel misalignment (see Appx. D.2).
        :param train_length: Input length during training.
        :param current_length: Current input length.
        :return: Returns the max relative position for the calculation of the relative
                 positions vector. The max. of train is always equal to 1.
        """
        # get sampling rate ratio
        if train_length > current_length:
            sr_change = round(train_length / current_length)
        else:
            sr_change = 1 / round(current_length / train_length)

        # get step sizes (The third parameter of torch.linspace).
        train_step = 2.0 / (train_length - 1)
        current_step = train_step * sr_change

        # Calculate the maximum relative position.
        if sr_change > 1:
            substract = (train_length - 1) % sr_change
            max_relative_pos = 1 - substract * train_step
        else:
            add = (current_length - 1) % (1 / sr_change)
            max_relative_pos = 1 + add * current_step
        return max_relative_pos


class CKBlock(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernelnet_hidden_channels: int,
            kernelnet_activation_function: str,
            kernelnet_norm_type: str,
            dim_linear: int,
            bias: bool,
            omega_0: bool,
            dropout: float,
            weight_dropout: float,
    ):
        """
        Creates a Residual Block with CKConvs as:
        ( Follows the Residual Block of Bai et. al., 2017 )
        input
         | ---------------|
         CKConv           |
         LayerNorm        |
         ReLU             |
         DropOut          |
         |                |
         CKConv           |
         LayerNorm        |
         ReLU             |
         DropOut          |
         + <--------------|
         |
         ReLU
         |
         output
        :param in_channels:  Number of channels in the input signal
        :param out_channels:  Number of output (and hidden) channels of the block.
        :param kernelnet_hidden_channels: Number of hidden units in the KernelNets of the CKConvs.
        :param kernelnet_activation_function: Activation function used in the KernelNets of the CKConvs.
        :param kernelnet_norm_type: Normalization type used in the KernelNets of the CKConvs (only for non-Sine KernelNets).
        :param dim_linear:  Spatial dimension of the input, e.g., for audio = 1, images = 2 (only 1 suported).
        :param bias:  If True, adds a learnable bias to the output.
        :param omega_0: Value of the omega_0 value of the KernelNets. (only for non-Sine KernelNets).
        :param dropout: Dropout rate of the block
        :param weight_dropout: Dropout rate applied to the sampled convolutional kernels.
        """
        super().__init__()

        # CKConv layers
        self.cconv1 = CKConv(
            in_channels,
            out_channels,
            kernelnet_hidden_channels,
            kernelnet_activation_function,
            kernelnet_norm_type,
            dim_linear,
            bias,
            omega_0,
            weight_dropout,
        )
        self.cconv2 = CKConv(
            out_channels,
            out_channels,
            kernelnet_hidden_channels,
            kernelnet_activation_function,
            kernelnet_norm_type,
            dim_linear,
            bias,
            omega_0,
            weight_dropout,
        )
        # Norm layers
        self.norm1 = LayerNorm(out_channels)
        self.norm2 = LayerNorm(out_channels)

        # Dropout
        self.dp = torch.nn.Dropout(dropout)

        shortcut = []
        if in_channels != out_channels:
            shortcut.append(Linear1d(in_channels, out_channels))
        self.shortcut = torch.nn.Sequential(*shortcut)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.dp(torch.relu(self.norm1(self.cconv1(x))))
        out = torch.relu(self.dp(torch.relu(self.norm2(self.cconv2(out)))) + shortcut)
        return out


class CKCNN(torch.nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            num_blocks: int,  # 2
            kernelnet_hidden_channels: int,
            kernelnet_activation_function: str,
            kernelnet_norm_type: str,
            dim_linear: int,
            bias: bool,
            omega_0: bool,  # sensitive to this param: good values <= 70
            dropout: float,
            weight_dropout: float,
            pool: bool,  # Always False in our experiments.
    ):
        super(CKCNN, self).__init__()

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                CKBlock(
                    # block_in_channels,
                    hidden_channels,
                    hidden_channels,
                    kernelnet_hidden_channels,
                    kernelnet_activation_function,
                    kernelnet_norm_type,
                    dim_linear,
                    bias,
                    omega_0,
                    dropout,
                    weight_dropout,
                )
            )
            if pool:
                blocks.append(torch.nn.MaxPool1d(kernel_size=2))
        self.backbone = torch.nn.Sequential(*blocks)

    def forward(self, x, *args, **kwargs):
        # Change from (B, L, H) -> (B, H, L)
        x = x.transpose(1, 2)
        x = self.backbone(x)
        x = x.transpose(1, 2)
        return x


class CopyMemory_CKCNN(CKCNN):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            num_blocks: int,
            kernelnet_hidden_channels: int,
            kernelnet_activation_function: str,
            kernelnet_norm_type: str,
            dim_linear: int,
            bias: bool,
            omega_0: bool,
            dropout: float,
            weight_dropout: float,
            pool: bool,
    ):
        super().__init__(
            in_channels,
            hidden_channels,
            num_blocks,
            kernelnet_hidden_channels,
            kernelnet_activation_function,
            kernelnet_norm_type,
            dim_linear,
            bias,
            omega_0,
            dropout,
            weight_dropout,
            pool,
        )

        self.finallyr = torch.nn.Linear(in_features=hidden_channels, out_features=10)
        # Initialize finallyr
        self.finallyr.weight.data.normal_(
            mean=0.0,
            std=0.01,
        )
        self.finallyr.bias.data.fill_(value=0.0)

    def forward(self, x, *args, **kwargs):
        # Change from (B, S, C) -> (B, C, S)
        x = x.transpose(1, 2)
        out = self.backbone(x)
        out = self.finallyr(out.transpose(1, 2))
        return out


class AddProblem_CKCNN(CKCNN):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            num_blocks: int,
            kernelnet_hidden_channels: int,
            kernelnet_activation_function: str,
            kernelnet_norm_type: str,
            dim_linear: int,
            bias: bool,
            omega_0: bool,
            dropout: float,
            weight_dropout: float,
            pool: bool,
    ):
        super().__init__(
            in_channels,
            hidden_channels,
            num_blocks,
            kernelnet_hidden_channels,
            kernelnet_activation_function,
            kernelnet_norm_type,
            dim_linear,
            bias,
            omega_0,
            dropout,
            weight_dropout,
            pool,
        )

        self.finallyr = torch.nn.Linear(in_features=hidden_channels, out_features=1)
        # Initialize finallyr
        self.finallyr.weight.data.normal_(
            mean=0.0,
            std=0.01,
        )
        self.finallyr.bias.data.fill_(value=0.0)

    def forward(self, x, *args, **kwargs):
        # Change from (B, S, C) -> (B, C, S)
        x = x.transpose(1, 2)
        out = self.backbone(x)
        out = self.finallyr(out[:, :, -1])
        return out


class ClassificationCKCNN(CKCNN):
    def __init__(
            self,
            # d_input: int,
            # d_output: int,
            d_model: int,
            num_blocks: int,
            kernelnet_hidden_channels: int,
            kernelnet_activation_function: str,
            kernelnet_norm_type: str,
            dim_linear: int,
            bias: bool,
            omega_0: bool,
            dropout: float,
            weight_dropout: float,
            pool: bool,
            wd: float,
            # **kwargs,
    ):
        super().__init__(
            # d_input,
            d_model,
            num_blocks,
            kernelnet_hidden_channels,
            kernelnet_activation_function,
            kernelnet_norm_type,
            dim_linear,
            bias,
            omega_0,
            dropout,
            weight_dropout,
            pool,
        )
        self.d_model = d_model
        self.d_output = d_model

        self.wd = LnLoss(wd, 2)

    def forward(self, x, *args, **kwargs):
        # Change from (B, S, C) -> (B, C, S)
        x = x.transpose(1, 2)
        x = self.backbone(x)
        x = x.transpose(1, 2)
        return x, None # Have to return a state

    def loss(self):
        return self.wd.forward(model=self)

class LnLoss(torch.nn.Module):
    def __init__(
            self,
            weight_loss: float,
            norm_type: int,
    ):
        """
        Computes the Ln loss on the CKConv kernels in a CKCNN.
        :param weight_loss: Specifies the weight with which the loss will be summed to the total loss.
        :param norm_type: Type of norm, e.g., 1 = L1 loss, 2 = L2 loss, ...
        """
        super(LnLoss, self).__init__()
        self.weight_loss = weight_loss
        self.norm_type = norm_type

    def forward(
            self,
            model: CKConv,
    ):
        loss = 0.0
        # Go through modules that are instances of CKConvs and gather the sampled filters
        for m in model.modules():
            if not isinstance(m, CKConv):
                continue
            loss = loss + m.conv_kernel.norm(self.norm_type)
            loss = loss + m.bias.norm(self.norm_type)

        loss = self.weight_loss * loss
        return loss
