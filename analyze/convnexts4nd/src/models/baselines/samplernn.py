"""Implementation of SampleRNN model.

Paper: https://arxiv.org/abs/1612.07837
"""

import torch
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np

from src.models.baselines.lstm import TorchLSTM
from src.models.baselines.gru import TorchGRU
from src.models.sequence.base import SequenceModule
from src.models.sequence.modules.s4block import S4Block
from src.dataloaders.audio import mu_law_decode, linear_decode, q_zero

class StackedRNN(SequenceModule):
    """
    StackedRNN with skip connections:
        Input (d_model) -> RNN_1 (d_hidden) -> Linear (d_hidden, d_hidden) -> Output
        [Input, RNN_1] (d_model + d_hidden) -> RNN_2 (d_hidden) -> Linear (d_hidden, d_hidden) -> += Output
        [Input, RNN_2] (d_model + d_hidden) -> RNN_3 (d_hidden) -> Linear (d_hidden, d_hidden) -> += Output
    ...
    """

    @property
    def d_output(self):
        return self.d_model if self.output_linear else self.d_hidden

    def __init__(
        self,
        d_model,
        d_hidden,
        n_layers,
        learn_h0=False,
        rnn_type='gru',
        skip_connections=False,
        weight_norm=False,
        dropout=0.0,
        output_linear=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.learn_h0 = learn_h0
        self.skip_connections = skip_connections
        self.weight_norm = torch.nn.utils.weight_norm if weight_norm else lambda x: x

        self.output_linear = output_linear
        self.rnn_layers = torch.nn.ModuleList()
        self.lin_layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        self.rnn_type = rnn_type

        if rnn_type == 'lstm':
            RNN = TorchLSTM
        elif rnn_type == 'gru':
            RNN = TorchGRU
        else:
            raise ValueError('rnn_type must be lstm or gru')

        for i in range(n_layers):

            if i == 0:
                self.rnn_layers.append(
                    RNN(d_model=d_model, d_hidden=d_hidden, n_layers=1, learn_h0=learn_h0),
                )
            else:
                if skip_connections:
                    self.rnn_layers.append(
                        RNN(d_model=d_model + d_hidden, d_hidden=d_hidden, n_layers=1, learn_h0=learn_h0),
                    )
                else:
                    self.rnn_layers.append(
                        RNN(d_model=d_hidden, d_hidden=d_hidden, n_layers=1, learn_h0=learn_h0),
                    )

            if skip_connections:
                self.lin_layers.append(self.weight_norm(torch.nn.Linear(d_hidden, d_hidden)))
            else:
                self.lin_layers.append(torch.nn.Identity())

            if dropout > 0.0 and i < n_layers - 1:
                self.dropout_layers.append(torch.nn.Dropout(dropout))
            else:
                self.dropout_layers.append(torch.nn.Identity())

        if output_linear:
            self.output_layer = self.weight_norm(torch.nn.Linear(d_hidden, d_model))
        else:
            self.output_layer = torch.nn.Identity()

        # Apply weight norm to all the RNN layers
        for rnn in self.rnn_layers:
            # Find all Linear layers in the RNN
            for name, module in rnn.named_modules():
                if isinstance(module, torch.nn.Linear):
                    setattr(rnn, name, self.weight_norm(module))

        # Use orthogonal initialization for W_hn if using GRU (weight_hh_l[0])
        if rnn_type == 'gru':
            for rnn in self.rnn_layers:
                torch.nn.init.orthogonal_(rnn.weight_hh_l0[2 * d_hidden:].data)


    def default_state(self, *batch_shape, device=None):
        return [
            rnn.default_state(*batch_shape, device=device)
            for rnn in self.rnn_layers
        ]

    def forward(self, inputs, *args, state=None, **kwargs):

        outputs = inputs
        prev_states = [None] * len(self.rnn_layers) if state is None else state
        next_states = []
        out = 0.
        for rnn, prev_state, lin, dropout in zip(self.rnn_layers, prev_states, self.lin_layers, self.dropout_layers):
            # Run RNN on inputs
            outputs, state = rnn(outputs, prev_state)
            next_states.append(state)

            # If dropout, only apply to the outputs of RNNs that are not the last one (like torch's LSTM)
            outputs = dropout(outputs)

            z = lin(outputs)

            if self.skip_connections:
                # If skip connections, add the outputs of all the RNNs to the outputs
                out += z
                # Feed in the outputs of the previous RNN, and the original inputs to the next RNN
                outputs = torch.cat([outputs, inputs], dim=-1)
            else:
                out = z
                outputs = z

        out = self.output_layer(out)

        return out, next_states


class StackedRNNBaseline(SequenceModule):
    """Standard stacked RNN baseline in SampleRNN paper.

    Marked as the "one_tier" model in the codebase.
    https://github.com/soroushmehr/sampleRNN_ICLR2017/blob/master/models/one_tier/one_tier.py

    Discrete Input (Q_LEVELS) -->
    Embedding (EMB_SIZE) -->

    ----------- (start) this module implements the RNN + Linear Layers backbone -----------
    StackedRNN (N_RNN \in [5], FRAME_SIZE, DIM, LEARNED_H0, WEIGHT_NORM, SKIP_CONNECTIONS) -->
    Linear (DIM, DIM) + ReLU -->
    Linear (DIM, DIM) + ReLU -->
    Linear (DIM, DIM) + ReLU -->
    ----------- (end) this module implements the RNN + Linear Layers backbone -----------

    Linear (DIM, Q_LEVELS)
    """

    @property
    def d_output(self):
        return self.d_hidden

    def __init__(
        self,
        d_model,
        d_hidden,
        n_layers,
        learn_h0=False,
        rnn_type='gru',
        weight_norm=False,
        skip_connections=True,
        dropout=0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.learn_h0 = learn_h0
        self.weight_norm = weight_norm
        self.skip_connections = skip_connections
        self.rnn_type = rnn_type

        self.rnn = StackedRNN(
            d_model=d_model,
            d_hidden=d_hidden,
            n_layers=n_layers,
            rnn_type=rnn_type,
            skip_connections=skip_connections,
            weight_norm=weight_norm,
            dropout=dropout,
            output_linear=False,
        )

        self.lin1 = torch.nn.Linear(d_hidden, d_hidden)
        self.lin2 = torch.nn.Linear(d_hidden, d_hidden)
        self.lin3 = torch.nn.Linear(d_hidden, d_hidden)

        if weight_norm:
            self.lin1 = torch.nn.utils.weight_norm(self.lin1)
            self.lin2 = torch.nn.utils.weight_norm(self.lin2)
            self.lin3 = torch.nn.utils.weight_norm(self.lin3)

    def default_state(self, *batch_shape, device=None):
        return self.rnn.default_state(*batch_shape, device=device)

    def forward(self, inputs, *args, state=None, **kwargs):
        outputs = inputs
        outputs, state = self.rnn(outputs, state)
        outputs = F.relu(self.lin1(outputs))
        outputs = F.relu(self.lin2(outputs))
        outputs = F.relu(self.lin3(outputs))

        return outputs, state


class LearnedUpsampling1d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()

        self.conv_t = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.FloatTensor(out_channels, kernel_size)
            )
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_t.reset_parameters()
        torch.nn.init.constant(self.bias, 0)

    def forward(self, input):
        (batch_size, _, length) = input.size()
        (kernel_size,) = self.conv_t.kernel_size
        bias = self.bias.unsqueeze(0).unsqueeze(2).expand(
            batch_size, self.conv_t.out_channels, length, kernel_size
        ).contiguous().view(
            batch_size, self.conv_t.out_channels,
            length * kernel_size
        )
        return self.conv_t(input) + bias


class SampleRNN(SequenceModule):
    """SampleRNN model.

    Implementation adapted from https://github.com/deepsound-project/samplernn-pytorch.
    """

    @property
    def d_output(self):
        return self.d_hidden

    def __init__(
        self,
        frame_sizes=(16, 4),
        n_rnn=2,
        d_hidden=1024,
        bits=8,
        learn_h0=True,
        d_model=256,
        weight_norm=True,
        reproduce=True,
        quantization='linear',
        layer='gru',
    ):
        super().__init__()

        self.d_hidden = d_hidden
        self.d_model = d_model
        self.reproduce = reproduce
        self.bits = bits
        self.quantization = quantization
        self.layer = layer

        if self.quantization == 'linear':
            self.dequantizer = linear_decode
        elif self.quantization == 'mu-law':
            self.dequantizer = mu_law_decode
        else:
            raise ValueError(f"Unknown quantization type: {self.quantization}")

        if not self.reproduce:
            self.encoder = torch.nn.Embedding(1 << bits, d_model)

        ns_frame_samples = map(int, np.cumprod(frame_sizes))  # e.g. (16, 4) -> (16, 64)
        self.frame_level_rnns = torch.nn.ModuleList([
            FrameLevelRNN(
                frame_size=frame_size,
                n_frame_samples=n_frame_samples,
                d_model=d_model,
                n_rnn=n_rnn,
                d_hidden=d_hidden,
                learn_h0=learn_h0,
                weight_norm=weight_norm,
                reproduce=reproduce,
                layer=layer,
            )
            for (frame_size, n_frame_samples) in zip(frame_sizes, ns_frame_samples)
        ])

        self.sample_level_mlp = SampleLevelMLP(
            frame_size=frame_sizes[0],
            d_hidden=d_hidden,
            bits=bits,
            d_model=d_model,
            weight_norm=weight_norm,
            reproduce=reproduce,
        )

    def default_state(self, batch_size, device=None):
        self._reset_state=True # Special hacks for SampleRNN
        return [rnn.default_state(batch_size, device=device) for rnn in self.frame_level_rnns]

    def step(self, x, state=None, *args, **kwargs):
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        batch_size = x.shape[0]

        assert state is not None, "SampleRNN: State should be constructed with default_state before forward pass"
        if self._reset_state: # Hacks for SampleRNN
            self._reset_state = False
            # state = self.default_state(batch_size, device=x.device)
            self._frame_level_outputs = [None for _ in self.frame_level_rnns]
            self._window = torch.zeros(
                batch_size,
                self.lookback,
                x.shape[1] if len(x.shape) == 2 else x.shape[2],
                dtype=x.dtype,
                device=x.device,
            ) + q_zero(bits=self.bits)
            self._step_idx = self.lookback

            if len(x.shape) == 3:
                assert x.shape[1] == self.lookback
                self._window = x

        if self._step_idx > self.lookback:
            # Update window (but on the first step)
            self._window[:, :-1] = self._window[:, 1:].clone()
            self._window[:, -1] = x

        new_states = []

        for (i, rnn), state_ in zip(reversed(list(enumerate(self.frame_level_rnns))), reversed(state)):
            if self._step_idx % rnn.n_frame_samples != 0:
                # Don't need to process this rnn
                new_states.append(state_)
                continue

            # prev_samples shape: (B, CHUNK_SIZE, D) e.g. (16, 16384, 1)
            prev_samples = self._window[:, -rnn.n_frame_samples:]

            if self.reproduce:
                # SampleRNN dequantizes to recover the raw audio signal before passing this to the RNN
                prev_samples = self.dequantizer(prev_samples, bits=self.bits)
                prev_samples = 2 * prev_samples.contiguous()
                # Below, reshape from (B, CHUNK_SIZE, D) -> (B, -1, rnn.n_frame_samples) = (B, M_i, F_i)
                # e.g. (16, 16384, 1) -> (16, 256, 64) [first rnn] | (16, 1024, 16) [second rnn]
                prev_samples = prev_samples.view(batch_size, -1, rnn.n_frame_samples)

            else:
                raise NotImplementedError
                # More generally, we can use an Embedding encoder instead
                prev_samples = self.encoder(prev_samples)
                prev_samples = prev_samples.contiguous()
                prev_samples = prev_samples.view(batch_size, -1, rnn.n_frame_samples, self.d_model)

            # upper_tier_conditioning shape: None -> (B, M, D_HIDDEN) [first rnn]
            # (B, M_{i-1}, D_HIDDEN) -> (B, M_i, D_HIDDEN) [second rnn]
            if i == len(self.frame_level_rnns) - 1:
                upper_tier_conditioning = None
            else:
                frame_index = (self._step_idx // rnn.n_frame_samples) % self.frame_level_rnns[i + 1].frame_size
                upper_tier_conditioning = self._frame_level_outputs[i + 1][:, frame_index, :].unsqueeze(1)

            upper_tier_conditioning, new_state = rnn(prev_samples, upper_tier_conditioning, state_)

            self._frame_level_outputs[i] = upper_tier_conditioning

            new_states.append(new_state)

        # Make sure new states are in the right order
        new_states = list(reversed(new_states))

        bottom_frame_size = self.frame_level_rnns[0].frame_size
        mlp_input_sequences = self._window[:, -bottom_frame_size:]

        # Upper tier conditioning for the bottom
        upper_tier_conditioning = self._frame_level_outputs[0][:, self._step_idx % bottom_frame_size, :].unsqueeze(1)
        y = self.sample_level_mlp(mlp_input_sequences, upper_tier_conditioning)

        # Update window and step
        self._step_idx += 1

        # mlp_input_sequences shape: (B, L - _, D) e.g. (16, 16399, 1)
        # upper_tier_conditioning shape: (B, M_{last_rnn}, D_HIDDEN) [last rnn]
        return y.squeeze(1), new_states # (B, D)

    @property
    def lookback(self):
        return self.frame_level_rnns[-1].n_frame_samples

    def forward(self, inputs, *args, state=None, **kwargs):
        """
        inputs shape: (B, L, D) e.g. (16, 16447, 1)

        For SampleRNN, inputs contains quantized audio samples (e.g. B elements of length L)
        """
        batch_size = inputs.shape[0]

        assert state is not None, "SampleRNN: State should be constructed with default_state before forward pass"

        upper_tier_conditioning = None
        new_states = []
        for rnn, state_ in zip(reversed(self.frame_level_rnns), reversed(state)):
            # TODO: explain this
            from_index = self.lookback - rnn.n_frame_samples
            to_index = -rnn.n_frame_samples + 1

            # prev_samples shape: (B, CHUNK_SIZE, D) e.g. (16, 16384, 1)
            prev_samples = inputs[:, from_index : to_index]

            if self.reproduce:
                # SampleRNN dequantizes to recover the raw audio signal before passing this to the RNN
                prev_samples = self.dequantizer(prev_samples, bits=self.bits)
                prev_samples = 2 * prev_samples.contiguous()
                # Below, reshape from (B, CHUNK_SIZE, D) -> (B, -1, rnn.n_frame_samples) = (B, M_i, F_i)
                # e.g. (16, 16384, 1) -> (16, 256, 64) [first rnn] | (16, 1024, 16) [second rnn]
                prev_samples = prev_samples.view(batch_size, -1, rnn.n_frame_samples)

            else:
                # More generally, we can use an Embedding encoder instead
                prev_samples = self.encoder(prev_samples)
                prev_samples = prev_samples.contiguous()
                prev_samples = prev_samples.view(batch_size, -1, rnn.n_frame_samples, self.d_model)

            # upper_tier_conditioning shape: None -> (B, M, D_HIDDEN) [first rnn]
            # (B, M_{i-1}, D_HIDDEN) -> (B, M_i, D_HIDDEN) [second rnn]
            upper_tier_conditioning, new_state = rnn(prev_samples, upper_tier_conditioning, state_)

            new_states.append(new_state)

        # Make sure new states are in the right order
        new_states = list(reversed(new_states))

        bottom_frame_size = self.frame_level_rnns[0].frame_size
        mlp_input_sequences = inputs[:, self.lookback - bottom_frame_size : ]

        # mlp_input_sequences shape: (B, L - _, D) e.g. (16, 16399, 1)
        # upper_tier_conditioning shape: (B, M_{last_rnn}, D_HIDDEN) [last rnn]
        return self.sample_level_mlp(mlp_input_sequences, upper_tier_conditioning), new_states

def lecun_uniform(tensor):
    fan_in = torch.nn.init._calculate_correct_fan(tensor, 'fan_in')
    torch.nn.init.uniform(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

def concat_init(tensor, inits):
    try:
        tensor = tensor.data
    except AttributeError:
        pass

    (length, fan_out) = tensor.size()
    fan_in = length // len(inits)

    chunk = tensor.new(fan_in, fan_out)
    for (i, init) in enumerate(inits):
        init(chunk)
        tensor[i * fan_in : (i + 1) * fan_in, :] = chunk

class FrameLevelRNN(torch.nn.Module):

    def __init__(
        self,
        frame_size,
        n_frame_samples,
        d_model,
        n_rnn,
        d_hidden,
        learn_h0=True,
        weight_norm=True,
        reproduce=False,
        layer='gru',
    ):
        super().__init__()

        self.frame_size = frame_size
        self.n_frame_samples = n_frame_samples
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_rnn = n_rnn
        self.learn_h0 = learn_h0
        self.weight_norm = weight_norm
        self.reproduce = reproduce
        self.layer = layer

        if self.reproduce:
            assert learn_h0, "Original SampleRNN FrameLevelRNN learns h0."
            assert weight_norm, "Original SampleRNN FrameLevelRNN uses weight norm."

        if reproduce:
            self.input_expand = torch.nn.Conv1d(
                in_channels=n_frame_samples,
                out_channels=d_hidden,
                kernel_size=1,
            )
            torch.nn.init.kaiming_uniform(self.input_expand.weight)
            torch.nn.init.constant(self.input_expand.bias, 0)
        else:
            self.input_expand = torch.nn.Conv1d(
                in_channels=d_model,
                out_channels=d_hidden,
                kernel_size=n_frame_samples,
                stride=n_frame_samples,
            )

        if self.layer == 'gru':
            self.rnn = TorchGRU(
                d_model=d_hidden,
                d_hidden=d_hidden,
                n_layers=n_rnn,
                learn_h0=learn_h0,
            )
        elif self.layer == 's4':
            self.rnn = S4Block(
                H=d_hidden,
                d_state=64,
                use_state=False,
            )

        if reproduce:

            if self.layer == 'gru':
                for i in range(n_rnn):
                    concat_init(
                        getattr(self.rnn, 'weight_ih_l{}'.format(i)),
                        [lecun_uniform, lecun_uniform, lecun_uniform]
                    )
                    torch.nn.init.constant(getattr(self.rnn, 'bias_ih_l{}'.format(i)), 0)

                    concat_init(
                        getattr(self.rnn, 'weight_hh_l{}'.format(i)),
                        [lecun_uniform, lecun_uniform, torch.nn.init.orthogonal]
                    )
                    torch.nn.init.constant(getattr(self.rnn, 'bias_hh_l{}'.format(i)), 0)

            self.upsampling = LearnedUpsampling1d(
                in_channels=d_hidden,
                out_channels=d_hidden,
                kernel_size=frame_size,
            )

            torch.nn.init.uniform(
                self.upsampling.conv_t.weight, -np.sqrt(6 / d_hidden), np.sqrt(6 / d_hidden)
            )
            torch.nn.init.constant(self.upsampling.bias, 0)
        else:
            self.upsampling = torch.nn.ConvTranspose1d(
                in_channels=d_hidden,
                out_channels=d_hidden,
                kernel_size=frame_size,
                stride=frame_size,
                bias=True,
            )


        if weight_norm and reproduce:
            self.input_expand = torch.nn.utils.weight_norm(self.input_expand)
            self.upsampling.conv_t = torch.nn.utils.weight_norm(self.upsampling.conv_t)
        else:
            self.input_expand = torch.nn.utils.weight_norm(self.input_expand)
            self.upsampling = torch.nn.utils.weight_norm(self.upsampling)

    def default_state(self, batch_size, device=None):
        if self.layer == 'gru':
            return self.rnn.default_state(batch_size, device=device)
        elif self.layer == 's4':
            return None

    def forward(self, prev_samples, upper_tier_conditioning, state=None):
        """
        prev_samples: (B, M_i, D_MODEL) if self.reproduce else (B, M_i, FRAME, D_MODEL)
        upper_tier_conditioning: (B, M_i, D_HIDDEN) or None
        """
        if not self.reproduce:
            # Use strided convolutions to get frame embeddings
            # This generalizes the SampleRNN operation to handle non-1D signals
            # This reshapes from (B, M_i, FRAME, D_MODEL) -> (B, M_i, D_HIDDEN)
            prev_samples = prev_samples.view(prev_samples.shape[0], -1, self.d_model)
            input = self.input_expand(prev_samples.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            # SampleRNN uses an MLP (implemented as 1D Conv) to map (FRAME_SIZE, 1) to D_HIDDEN
            # This reshapes from (B, M_i, FRAME) -> (B, M_i, D_HIDDEN)
            input = self.input_expand(prev_samples.permute(0, 2, 1)).permute(0, 2, 1)

        if upper_tier_conditioning is not None:
            input += upper_tier_conditioning

        # Run RNN: (B, M_i, D_HIDDEN) -> (B, M_i, D_HIDDEN)
        if self.layer == 'gru':
            output, state = self.rnn(input, state.contiguous())
        elif self.layer == 's4':
            # TODO: not working
            output, state = self.rnn(input.transpose(1, 2), state)
            output = output.transpose(1, 2)

        # Run 1D transposed convolution to upsample: (B, M_i, D_HIDDEN) -> (B, M', D_HIDDEN)
        # TODO: make M' more precise
        output = self.upsampling(output.permute(0, 2, 1)).permute(0, 2, 1)

        return output, state


class SampleLevelMLP(torch.nn.Module):

    def __init__(
        self,
        frame_size,
        d_hidden,
        bits=8,
        d_model=256,
        weight_norm=True,
        embedding=True,
        reproduce=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.reproduce = reproduce

        if self.reproduce:
            assert embedding, "Original SampleRNN SampleLevelMLP uses an embedding layer."
            assert weight_norm, "Original SampleRNN SampleLevelMLP uses weight norm."

        if embedding:
            self.embedding = torch.nn.Embedding(1 << bits, d_model)

        self.input = torch.nn.Conv1d(
            in_channels=d_model,
            out_channels=d_hidden,
            kernel_size=frame_size,
            bias=False,
        )

        if self.reproduce:
            self.hidden = torch.nn.Conv1d(
                in_channels=d_hidden,
                out_channels=d_hidden,
                kernel_size=1,
            )
        else:
            self.hidden = torch.nn.Linear(d_hidden, d_hidden)

        if self.reproduce:
            self.output = torch.nn.Conv1d(
                in_channels=d_hidden,
                out_channels=256,
                kernel_size=1,
            )
        else:
            self.output = torch.nn.Linear(d_hidden, 256)

        if self.reproduce:
            torch.nn.init.kaiming_uniform(self.input.weight)
            torch.nn.init.kaiming_uniform(self.hidden.weight)
            torch.nn.init.constant(self.hidden.bias, 0)
            lecun_uniform(self.output.weight)
            torch.nn.init.constant(self.output.bias, 0)

        if weight_norm:
            self.input = torch.nn.utils.weight_norm(self.input)
            self.hidden = torch.nn.utils.weight_norm(self.hidden)
            self.output = torch.nn.utils.weight_norm(self.output)

    def forward(self, prev_samples, upper_tier_conditioning):
        if self.embedding:
            # Embed the input samples (which are quantized)
            # This reshapes from (B, L, 1) -> (B, L, D_MODEL)
            prev_samples = self.embedding(
                prev_samples.contiguous().view(-1)
            ).view(prev_samples.shape[0], -1, self.d_model)

        assert prev_samples.shape[-1] == self.d_model, "`prev_samples` shape should be (B, L', D_MODEL)"

        # prev_samples: (B, L', D_MODEL) -> (B, D_MODEL, L')
        # upper_tier_conditioning: (B, L, D_HIDDEN) -> (B, D_HIDDEN, L)
        prev_samples = prev_samples.permute(0, 2, 1)
        upper_tier_conditioning = upper_tier_conditioning.permute(0, 2, 1)

        if self.reproduce:
            # Take (B, L', D_MODEL), (B, L, D_HIDDEN) -> (B, D_HIDDEN, L)
            x = F.relu(self.input(prev_samples) + upper_tier_conditioning)
            x = F.relu(self.hidden(x))
            x = self.output(x).permute(0, 2, 1)
        else:
            # Take (B, L', D_MODEL), (B, L, D_HIDDEN) -> (B, D_HIDDEN, L)
            x = F.relu(self.input(prev_samples) + upper_tier_conditioning)
            # x: (B, D_HIDDEN, L) -> (B, L, D_HIDDEN)
            x = x.permute(0, 2, 1)
            x = F.relu(self.hidden(x))
            x = self.output(x)

        return x.contiguous()
