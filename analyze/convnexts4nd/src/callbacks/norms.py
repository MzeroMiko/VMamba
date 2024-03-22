import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.parsing import AttributeDict
from omegaconf import OmegaConf

class TrackNorms(pl.Callback):

    # TODO do callbacks happen before or after the method in the main LightningModule?
    # @rank_zero_only # needed?
    def on_after_training_step(self, batch, batch_idx, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Log extra metrics
        metrics = {}

        if hasattr(pl_module, "_grad_norms"):
            metrics.update(pl_module._grad_norms)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )


    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # example to inspect gradient information in tensorboard
        if OmegaConf.select(trainer.hparams, 'train.track_grad_norms'): # TODO dot notation should work with omegaconf?
            norms = {}
            for name, p in pl_module.named_parameters():
                if p.grad is None:
                    continue

                # param_norm = float(p.grad.data.norm(norm_type))
                param_norm = torch.mean(p.grad.data ** 2)
                norms[f"grad_norm.{name}"] = param_norm
            pl_module._grad_norms = norms

