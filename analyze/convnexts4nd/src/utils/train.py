"""Utils for the training loop. Adapted from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py."""
import logging
import os
import warnings
from typing import List, Sequence

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from src.utils.config import omegaconf_filter_keys


# Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
# [21-09-17 AG] copied from Tri's infra, doesn't appear to be used
class LoggingContext:
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def process_config(config: DictConfig) -> DictConfig: # TODO because of filter_keys, this is no longer in place
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    log = get_logger()

    OmegaConf.register_new_resolver('eval', eval)

    # Filter out keys that were used just for interpolation
    # config = dictconfig_filter_keys(config, lambda k: not k.startswith('__'))
    config = omegaconf_filter_keys(config, lambda k: not k.startswith('__'))

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

        # force debugger friendly configuration
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.loader.get("pin_memory"):
            config.loader.pin_memory = False
        if config.loader.get("num_workers"):
            config.loader.num_workers = 0

    # disable adding new keys to config
    # OmegaConf.set_struct(config, True) # [21-09-17 AG] I need this for .pop(_name_) pattern among other things

    return config

@rank_zero_only
def print_config(
    config: DictConfig,
    # fields: Sequence[str] = (
    #     "trainer",
    #     "model",
    #     "datamodule",
    #     "train",
    #     "callbacks",
    #     "logger",
    #     "seed",
    # ),
    resolve: bool = True,
    save_cfg=True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    if save_cfg:
        with open("config_tree.txt", "w") as fp:
            rich.print(tree, file=fp)

def log_optimizer(logger, optimizer, keys):
    """ Log values of particular keys from the optimizer's param groups """
    keys = sorted(keys)
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        n_params = sum(p.numel() for p in g['params'])
        logger.info(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
            f"{n_params} parameters",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))
        # print(f"Optimizer group {i} | {len(g['params'])} tensors | lr {g['lr']} | wd {g.get('weight_decay', None)}")

"""Old code for resuming logic moved to tests/"""
