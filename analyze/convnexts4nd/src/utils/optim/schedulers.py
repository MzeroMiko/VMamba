"""Custom learning rate schedulers."""

import math
import warnings
import torch

from timm.scheduler import CosineLRScheduler


# https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
class CosineWarmup(torch.optim.lr_scheduler.CosineAnnealingLR):

    def __init__(self, optimizer, T_max, eta_min=0, warmup_step=0, **kwargs):
        self.warmup_step = warmup_step
        super().__init__(optimizer, T_max - warmup_step, eta_min, *kwargs)

    # Copied from CosineAnnealingLR, but adding warmup and changing self.last_epoch to
    # self.last_epoch - self.warmup_step.
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == self.warmup_step:  # also covers the case where both are 0
            return self.base_lrs
        elif self.last_epoch < self.warmup_step:
            return [base_lr * (self.last_epoch + 1) / self.warmup_step for base_lr in self.base_lrs]
        elif (self.last_epoch - self.warmup_step - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.warmup_step) / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_step - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    _get_closed_form_lr = None


def InvSqrt(optimizer, warmup_step):
    """Originally used for Transformer (in "Attention is All You Need")."""

    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == warmup_step:  # also covers the case where both are 0
            return 1.
        else:
            return 1. / (step ** 0.5) if step > warmup_step else (step + 1) / (warmup_step ** 1.5)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def Constant(optimizer, warmup_step):

    def lr_lambda(step):
        if step == warmup_step:  # also covers the case where both are 0
            return 1.
        else:
            return 1. if step > warmup_step else (step + 1) / warmup_step

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class TimmCosineLRScheduler(CosineLRScheduler, torch.optim.lr_scheduler._LRScheduler):
    """Wrap timm.scheduler.CosineLRScheduler so we can call scheduler.step() without passing in epoch.
    It supports resuming as well.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_epoch = -1
        self.step(epoch=0)

    def step(self, epoch=None):
        if epoch is None:
            self._last_epoch += 1
        else:
            self._last_epoch = epoch
        # We call either step or step_update, depending on whether we're using the scheduler every
        # epoch or every step.
        # Otherwise, lightning will always call step (i.e., meant for each epoch), and if we set
        # scheduler interval to "step", then the learning rate update will be wrong.
        if self.t_in_epochs:
            super().step(epoch=self._last_epoch)
        else:
            super().step_update(num_updates=self._last_epoch)
