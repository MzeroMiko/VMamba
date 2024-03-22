"""Log parameter counts to WandB."""

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.parsing import AttributeDict


class ParamsLog(pl.Callback):
    """ Log the number of parameters of the model """
    def __init__(
        self,
        total: bool = True,
        trainable: bool = True,
        fixed: bool = True,
    ):
        super().__init__()
        self._log_stats = AttributeDict(
            {
                'total_params_log': total,
                'trainable_params_log': trainable,
                'non_trainable_params_log': fixed,
            }
        )

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logs = {}
        if self._log_stats.total_params_log:
            logs["params/total"] = sum(p.numel() for p in pl_module.parameters())
        if self._log_stats.trainable_params_log:
            logs["params/trainable"] = sum(p.numel() for p in pl_module.parameters()
                                             if p.requires_grad)
        if self._log_stats.non_trainable_params_log:
            logs["params/fixed"] = sum(p.numel() for p in pl_module.parameters()
                                                     if not p.requires_grad)
        if trainer.logger:
            trainer.logger.log_hyperparams(logs)
