"""Wrapper of optimizers in torch.optim for computation of exponential moving average of parameters.

Source: https://github.com/kamenbliznashki/pixel_models/blob/master/optim.py
"""

import torch

def build_ema_optimizer(optimizer_cls):
    class Optimizer(optimizer_cls):
        def __init__(self, *args, polyak=0.0, **kwargs):
            if not 0.0 <= polyak <= 1.0:
                raise ValueError("Invalid polyak decay rate: {}".format(polyak))
            super().__init__(*args, **kwargs)
            self.defaults['polyak'] = polyak
            self.stepped = False

        def step(self, closure=None):
            super().step(closure)
            self.stepped = True

            # update exponential moving average after gradient update to parameters
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]

                    # state initialization
                    if 'ema' not in state:
                        state['ema'] = p.data.clone() # torch.zeros_like(p.data)

                    # ema update
                    state['ema'] -= (1 - self.defaults['polyak']) * (state['ema'] - p.data)


        def swap_ema(self):
            """ substitute exponential moving average values into parameter values """
            for group in self.param_groups:
                for p in group['params']:
                    data = p.data
                    state = self.state[p]
                    p.data = state['ema']
                    state['ema'] = data

        def __repr__(self):
            s = super().__repr__()
            return self.__class__.__mro__[1].__name__ + ' (\npolyak: {}\n'.format(self.defaults['polyak']) + s.partition('\n')[2]

    Optimizer.__name__ = optimizer_cls.__name__
    return Optimizer

Adam = build_ema_optimizer(torch.optim.Adam)
RMSprop = build_ema_optimizer(torch.optim.RMSprop)
