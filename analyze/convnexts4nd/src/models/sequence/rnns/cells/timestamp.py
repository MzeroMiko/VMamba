"""Variants of the HiPPO-RNN that accept timestamped inputs and evolve according to the elapsed time between inputs. Used in original HiPPO paper for irregularly-sampled CharacterTrajectories experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from src.models.sequence.rnns.cells.memory import MemoryCell, forward_aliases, backward_aliases, bilinear_aliases, zoh_aliases
from src.models.hippo.transition import (
    LegSAdaptiveTransitionManual,
    LegTAdaptiveTransitionManual,
    LagTAdaptiveTransitionManual,
    LegSTriDInverseAdaptiveTransition,
    LegTTriDInverseAdaptiveTransition,
    LagTTriDInverseAdaptiveTransition,
)

class TimeMemoryCell(MemoryCell):
    """MemoryCell with timestamped data.

    Assumes that first channel of inputs are timestamps.
    """

    def __init__(
            self,
            d_input, d_model, memory_size, memory_order,
            measure='legs',
            method='trid',
            discretization='bilinear',
            **kwargs
        ):
        if memory_order < 0:
            memory_order = d_model

        super().__init__(d_input-1, d_model, memory_size, memory_order, **kwargs)

        assert measure in ['legs', 'lagt', 'legt']
        assert method in ['dense', 'trid']
        transitions = {
            'dense': {
                'legs': LegSAdaptiveTransitionManual,
                'legt': LegTAdaptiveTransitionManual,
                'lagt': LagTAdaptiveTransitionManual,
            },
            'trid': {
                'legs': LegSTriDInverseAdaptiveTransition,
                'legt': LegTTriDInverseAdaptiveTransition,
                'lagt': LagTTriDInverseAdaptiveTransition,
            },
        }
        self.transition = transitions[method][measure](self.memory_order)

        if discretization in forward_aliases:
            self.transition_fn = partial(self.transition.forward_diff, **kwargs)
        elif discretization in backward_aliases:
            self.transition_fn = partial(self.transition.backward_diff, **kwargs)
        elif discretization in bilinear_aliases:
            self.transition_fn = partial(self.transition.bilinear, **kwargs)
        else: assert False

    def update_memory(self, m, u, t0, t1):
        """This class is intended to be subclassed to the LTI or LSI cases."""
        raise NotImplementedError

    def forward(self, input, state):
        h, m, prev_timestamp = state
        timestamp, input = input[:, 0], input[:, 1:]

        # Update the memory
        u = self.forward_memory(input, h, m)
        m = self.update_memory(m, u, prev_timestamp, timestamp) # (batch, memory_size, memory_order)

        # Update hidden
        h = self.forward_hidden(input, h, m)

        next_state = (h, m, timestamp)
        output = self.state_to_tensor(next_state)

        return output, next_state

class TimeLSICell(TimeMemoryCell):
    """A cell implementing "Linear Scale Invariant" dynamics: c' = Ac + Bf with timestamped inputs.

    This class can handle the setting where there is timescale shift, even if the model does not know about it.
    """

    name = 'tlsi'

    def update_memory(self, m, u, t0, t1):
        """
        m: (B, M, N) [batch, memory_size, memory_order]
        u: (B, M)
        t0: (B,) previous time
        t1: (B,) current time
        """

        if torch.eq(t1, 0.).any():
            return F.pad(u.unsqueeze(-1), (0, self.memory_order - 1))
        else:
            dt = ((t1-t0)/t1).unsqueeze(-1)
            m = self.transition_fn(dt, m, u)
        return m

class TimeLTICell(TimeMemoryCell):
    """A cell implementing Linear Time Invariant dynamics: c' = Ac + Bf with timestamped inputs.

    Unlike HiPPO-LegS with timestamps, this class will not work if there is timescale shift that it does not know about.
    However, unlike generic RNNs, it does work if it knows the sampling rate change.
    """

    name = 'tlti'

    def __init__(
            self,
            d_input, d_model, memory_size=1, memory_order=-1,
            measure='legt',
            dt=1.0,
            **kwargs
        ):
        if memory_order < 0:
            memory_order = d_model

        self.dt = dt

        super().__init__(d_input, d_model, memory_size, memory_order, measure=measure, **kwargs)

    def update_memory(self, m, u, t0, t1):
        """
        m: (B, M, N) [batch, memory_size, memory_order]
        u: (B, M)
        t0: (B,) previous time
        t1: (B,) current time
        """

        dt = self.dt*(t1-t0).unsqueeze(-1)
        m = self.transition_fn(dt, m, u)
        return m
