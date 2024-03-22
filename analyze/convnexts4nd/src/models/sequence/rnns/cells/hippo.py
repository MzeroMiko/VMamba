"""Implementation of full HiPPO-RNN variants."""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from src.models.sequence.rnns.cells.memory import LTICell, LSICell
from src.models.hippo.hippo import transition


class HiPPOLTICell(LTICell):
    measure = None

    def __init__(
            self, d_input, d_model, memory_size=1, memory_order=-1,
            measure_args={},
            **kwargs
        ):
        if memory_order < 0:
            memory_order = d_model

        A, B = transition(type(self).measure, memory_order, **measure_args)
        super().__init__(d_input, d_model, memory_size, memory_order, A, B, **kwargs)

class HiPPOLSICell(LSICell):
    measure = None

    def __init__(
            self, d_input, d_model, memory_size=1, memory_order=-1,
            measure_args={},
            **kwargs
        ):
        if memory_order < 0:
            memory_order = d_model

        A, B = transition(type(self).measure, memory_order, **measure_args)
        super().__init__(d_input, d_model, memory_size, memory_order, A, B, **kwargs)

class LegTCell(HiPPOLTICell):
    """Translated Legendre."""
    name = 'legt'
    measure = 'legt'

class LegSCell(HiPPOLSICell):
    """Scaled Legendre."""
    name = 'legs'
    measure = 'legs'

class LagTCell(HiPPOLTICell):
    """Translated Laguerre."""
    name = 'lagt'
    measure = 'lagt'

    def __init__(self, d_input, d_model, dt=1.0, **kwargs):
        super().__init__(d_input, d_model, dt=dt, **kwargs)

class GLagTCell(HiPPOLTICell):
    """Translated Generalized Laguerre."""
    name = 'glagt'
    measure = 'glagt'

    def __init__(self, d_input, d_model, dt=1.0, **kwargs):
        super().__init__(d_input, d_model, dt=dt, **kwargs)

class LMUCell(HiPPOLTICell):
    """This cell differs from the HiPPO-LegT cell by a normalization in the recurrent matrix A, and different RNN connections and initialization.

    https://papers.nips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf
    """
    name = 'lmu'
    measure = 'lmu'

    @property
    def default_initializers(self):
        return {
            'uxh': 'uniform',
            'ux': 'one',
            'uh': 'zero',
            'um': 'zero',
            'hxm': 'xavier',
            'hx': 'zero',
            'hh': 'zero',
            'hm': 'xavier',
        }

    @property
    def default_architecture(self):
        return {
            'ux': True,
            'um': True,
            'hx': True,
            'hm': True,
            'hh': True,
            'bias': False,
        }

    def __init__(self, d_input, d_model, theta=100, dt=1., gate='N', **kwargs):
        super().__init__(d_input, d_model, dt=dt/theta, gate=gate, **kwargs)


class RandomCell(LTICell):
    """ Ablation: demonstrate that random A matrix is not effective. """
    name = 'random'

    def __init__(
            self, d_input, d_model, memory_size=1, memory_order=-1,
            **kwargs
        ):
        if memory_order < 0:
            memory_order = d_model

        N = memory_order
        A = np.random.normal(size=(N, N)) / N**.5
        B = np.random.normal(size=(N, 1))

        super().__init__(d_input, d_model, memory_size, memory_order, A, B, **kwargs)
# TODO remove the noise cell, rename all the OP stuff into HiPPO
