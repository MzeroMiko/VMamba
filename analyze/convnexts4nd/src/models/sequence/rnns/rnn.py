import torch
import torch.nn as nn

import src.utils as utils
from src.models.sequence.rnns.cells import CellBase
from src.models.sequence import SequenceModule

# [21-09-12 AG]: We previously set up a way to register RNNCell classes, which gives them a "local" name
# To convert this mapping from name to constructor, we use the fact that the str representation of a constructor is "<class '_target_'>"
# TODO should convert this to an explicit dictionary
cell_registry = {
    name: str(target)[8:-2]
    for name, target in CellBase.registry.items()
}

class RNN(SequenceModule):
    def __init__(self, d_input, cell=None, return_output=True, transposed=False, dropout=0.0):
        """
        return_output: if False, only returns the state
        """
        super().__init__()
        self.transposed = transposed
        if dropout > 0.0:
            raise NotImplementedError("Dropout currently not supported for custom RNNs")
        self.return_output = return_output

        self.cell = utils.instantiate(cell_registry, cell, d_input)

    def forward(self, inputs, state=None, **kwargs):
        """
        cell.forward : (input, state) -> (output, state)
        inputs : [n_batch, l_seq, d]
        """

        if self.transposed: inputs = inputs.transpose(-1, -2)

        # Automatically detect PackedSequence
        if isinstance(inputs, nn.utils.rnn.PackedSequence):
            return PackedRNN.forward(self, inputs)

        # Construct initial state
        state = self.cell.default_state(*inputs.shape[:-2], device=inputs.device)
        outputs = []

        for input in torch.unbind(inputs, dim=-2):
            output, new_state = self.step(input, state)
            state = new_state
            if self.return_output:
                outputs.append(output)
        outputs = torch.stack(outputs, dim=-2) if self.return_output else None
        if self.transposed and outputs is not None: outputs = outputs.transpose(-1, -2)
        return outputs, state

    def step(self, x, state):
        return self.cell.step(x, state)

    def default_state(self, *args, **kwargs):
        return self.cell.default_state(*args, **kwargs)

    @property
    def d_state(self):
        """Size after converting state to a single tensor."""
        return self.cell.d_state

    @property
    def d_output(self):
        """Size of output."""
        return self.cell.d_output

    @property
    def state_to_tensor(self):
        """Convert state into a single tensor output."""
        return self.cell.state_to_tensor


class PackedRNN(RNN):
    """Version of RNN that expected a nn.utils.rnn.PackedSequence."""

    @staticmethod
    def apply_tuple(tup, fn):
        """Apply a function to a Tensor or a tuple of Tensor"""
        if isinstance(tup, tuple):
            return tuple((fn(x) if isinstance(x, torch.Tensor) else x) for x in tup)
        else:
            return fn(tup)

    @staticmethod
    def concat_tuple(tups, dim=0):
        """Concat a list of Tensors or a list of tuples of Tensor."""
        if isinstance(tups[0], tuple):
            return tuple(
                (torch.cat(xs, dim) if isinstance(xs[0], torch.Tensor) else xs[0])
                for xs in zip(*tups)
            )
        else:
            return torch.cat(tups, dim)

    def forward(self, inputs, len_batch=None):
        # assert len_batch is not None
        # inputs = nn.utils.rnn.pack_padded_sequence(
        #     inputs, len_batch.cpu(), enforce_sorted=False
        # )
        assert isinstance(inputs, nn.utils.rnn.PackedSequence)

        # Similar implementation to https://github.com/pytorch/pytorch/blob/9e94e464535e768ad3444525aecd78893504811f/torch/nn/modules/rnn.py#L202
        inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
        max_batch_size = batch_sizes[0]

        # Construct initial state
        state = self.cell.default_state(max_batch_size, device=inputs.device)
        outputs = []

        # Following implementation at https://github.com/pytorch/pytorch/blob/9e94e464535e768ad3444525aecd78893504811f/aten/src/ATen/native/RNN.cpp#L621
        # Batch sizes is a sequence of decreasing lengths, which are offsets
        # into a 1D list of inputs. At every step we slice out batch_size elements,
        # and possibly account for the decrease in the batch size since the last step,
        # which requires us to slice the hidden state (since some sequences
        # are completed now). The sliced parts are also saved, because we will need
        # to return a tensor of final hidden state.
        batch_sizes_og = batch_sizes
        batch_sizes = batch_sizes.detach().cpu().numpy()
        input_offset = 0
        last_batch_size = batch_sizes[0]
        saved_states = []
        for batch_size in batch_sizes:
            step_input = inputs[input_offset : input_offset + batch_size]
            input_offset += batch_size
            dec = last_batch_size - batch_size
            if dec > 0:
                saved_state = PackedRNN.apply_tuple(state, lambda x: x[batch_size:])
                state = PackedRNN.apply_tuple(state, lambda x: x[:batch_size])
                saved_states.append(saved_state)
            last_batch_size = batch_size
            output, new_state = self.cell.forward(step_input, state)
            state = new_state
            if self.return_output:
                outputs.append(output)
        saved_states.append(state)
        saved_states.reverse()
        state = PackedRNN.concat_tuple(saved_states)
        state = PackedRNN.apply_tuple(
            state,
            lambda x: x[unsorted_indices] if unsorted_indices is not None else x,
        )
        if self.return_output:
            outputs = nn.utils.rnn.PackedSequence(
                torch.cat(outputs, dim=0),
                batch_sizes_og,
                sorted_indices,
                unsorted_indices,
            )
            # outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        else:
            outputs = None
        return outputs, state
