"""Implements Task interface, which consists of encoder + decoder + loss/metrics."""

from typing import Optional, List, Tuple
import math
import functools
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import ListConfig
from src.models.nn.normalization import (
    ReversibleInstanceNorm1dInput,
    ReversibleInstanceNorm1dOutput,
    TSNormalization,
    TSInverseNormalization,
)

from src.models.nn.adaptive_softmax import AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
import src.tasks.metrics as M
import src.models.nn.utils as U
import torchmetrics as tm
from src.utils.config import to_list, instantiate


class BaseTask:
    """Abstract class for all tasks.

    This class takes care of:
    - loss function
    - arbitrary metrics
    - (optional) encoder module that interfaces with dataset (inputs) and model
    - (optional) decoder module that interfaces with dataset (targets) and model
    """
    encoder = None
    decoder = None

    def __init__(self, dataset=None, model=None, loss=None, loss_val=None, metrics=None, torchmetrics=None):
        """ This class is allowed to grab attributes directly off a constructed dataset and model object """
        self.dataset = dataset
        self.model = model
        if metrics is None: metrics = []
        self.metric_names = to_list(metrics)

        if torchmetrics is None: torchmetrics = []
        self.torchmetric_names = to_list(torchmetrics)
        self._tracked_torchmetrics = {}

        # The decoder might pass through arguments that the loss needs (e.g. sequence lengths)
        # but might also pass through extraneous arguments (e.g. sampling rate)
        # Wrap loss and metrics so that they accept kwargs and

        # Create loss function
        self.loss = instantiate(M.output_metric_fns, loss, partial=True)
        self.loss = U.discard_kwargs(self.loss)
        if loss_val is not None:
            self.loss_val = instantiate(M.output_metric_fns, loss_val, partial=True)
            self.loss_val = U.discard_kwargs(self.loss_val)

    def _init_torchmetrics(self, prefix):
        """Instantiate torchmetrics."""
        # TODO torchmetrics is better renamed to "epoch_metrics" or something

        self._tracked_torchmetrics[prefix] = {}
        for name in self.torchmetric_names:
            if name in ['AUROC', 'StatScores', 'Precision', 'Recall', 'F1', 'F1Score']:
                self._tracked_torchmetrics[prefix][name] = getattr(tm, name)(average='macro', num_classes=self.dataset.d_output, compute_on_step=False).to('cuda')
            elif '@' in name:
                k = int(name.split('@')[1])
                mname = name.split('@')[0]
                self._tracked_torchmetrics[prefix][name] = getattr(tm, mname)(average='macro', num_classes=self.dataset.d_output, compute_on_step=False, top_k=k).to('cuda')
            else:
                self._tracked_torchmetrics[prefix][name] = getattr(tm, name)(compute_on_step=False).to('cuda')

    def _reset_torchmetrics(self, prefix=None):
        """Reset torchmetrics for a prefix associated with a particular dataloader (e.g. train, val, test).

        Generally do this at the start of an epoch.
        """
        all_prefixes = [prefix] if prefix is not None else self._tracked_torchmetrics
        for prefix in all_prefixes:
            for name in self.torchmetric_names:
                try:
                    self._tracked_torchmetrics[prefix][name].reset()
                except KeyError:  # metrics don't exist yet
                    pass

    def get_torchmetrics(self, prefix):
        """Compute torchmetrics for a prefix associated with a particular dataloader (e.g. train, val, test).

        Generally do this at the end of an epoch.
        """
        return {name: self._tracked_torchmetrics[prefix][name].compute() for name in self.torchmetric_names}

    def torchmetrics(self, x, y, prefix):
        """Update torchmetrics with new data.

        Prefix corresponds to a particular dataloader (e.g. train, val, test).

        Generally call this every batch.
        """
        if prefix not in self._tracked_torchmetrics:
            self._init_torchmetrics(prefix)

        for name in self.torchmetric_names:
            if name.startswith('Accuracy'):
                if len(x.shape) > 2:
                    # Multi-dimensional, multi-class
                    self._tracked_torchmetrics[prefix][name].update(x.transpose(1, 2), y.squeeze())
                    continue
            self._tracked_torchmetrics[prefix][name].update(x, y)

    def metrics(self, x, y, **kwargs):
        """Add metrics to the task.

        Metrics are just functions:
        - output metrics are a function of output and target
        - loss metrics are a function of loss (e.g. perplexity)
        """
        output_metrics = {
            name: U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)
            for name in self.metric_names if name in M.output_metric_fns
        }
        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }
        return {**output_metrics, **loss_metrics}


class Scalar(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.c = c
    def forward(self, x):
        return x * self.c

class LMTask(BaseTask):
    def __init__(self, tied=False, rescale=True, **kwargs):
        super().__init__(loss='cross_entropy', **kwargs)
        n_tokens = self.dataset.n_tokens
        d_model = self.model.d_model
        d_output = self.model.d_output

        if rescale:
            scale = Scalar(math.sqrt(d_model))
        else:
            scale = None

        embedding = nn.Embedding(n_tokens, d_model)
        nn.init.normal_(embedding.weight, mean=0, std=d_model**-.5)
        encoder = U.PassthroughSequential(
            embedding,
            scale,
        )
        self.encoder = encoder
        decoder = nn.Linear(d_output, n_tokens)

        if tied:
            assert d_model == d_output
            decoder.weight = self.encoder[0].weight
        self.decoder = decoder

class ForecastingTask(BaseTask):

    class DummyModule(nn.Module):

        def forward(self, *args):
            return args

    def __init__(self, norm='mean', **kwargs):
        super().__init__(**kwargs)

        if norm == 'revnorm':
            self.encoder = ReversibleInstanceNorm1dInput(self.dataset.d_input, transposed=False)
            self.decoder = ReversibleInstanceNorm1dOutput(self.encoder)
        elif norm == 'mean':
            self.encoder = TSNormalization(method='mean', horizon=self.dataset.dataset_train.forecast_horizon)
            self.decoder = TSInverseNormalization(method='mean', normalizer=self.encoder)
        elif norm == 'last':
            self.encoder = TSNormalization(method='last', horizon=self.dataset.dataset_train.forecast_horizon)
            self.decoder = TSInverseNormalization(method='last', normalizer=self.encoder)
        else:
            self.encoder = None
            self.decoder = None

        try:
            if hasattr(self.dataset.dataset_train, 'mean'):
                self.mean = torch.tensor(self.dataset.dataset_train.mean)
                self.std = torch.tensor(self.dataset.dataset_train.std)
            elif hasattr(self.dataset.dataset_train, 'standardization'):
                self.mean = torch.tensor(self.dataset.dataset_train.standardization['means'])
                self.std = torch.tensor(self.dataset.dataset_train.standardization['stds'])
            else:
                self.mean = None
                self.std = None
        except AttributeError:
            raise AttributeError('Dataset does not have mean/std attributes')
            self.mean = torch.tensor(self.dataset.dataset_train.standardization['means'])
            self.std = torch.tensor(self.dataset.dataset_train.standardization['stds'])

        if hasattr(self.dataset.dataset_train, 'log_transform'):
            self.log_transform = self.dataset.dataset_train.log_transform
        else:
            self.log_transform = False
        print("Log Transform", self.log_transform)

    def metrics(self, x, y, state=None, timestamps=None, ids=None): # Explicit about which arguments the decoder might pass through, but can future-proof with **kwargs
        if self.mean is not None:
            means = self.mean[ids].to(x.device)
            stds = self.std[ids].to(x.device)
            x_ = x * stds[:, None, None] + means[:, None, None]
            y_ = y * stds[:, None, None] + means[:, None, None]
        else:
            x_ = x
            y_ = y

        if self.log_transform:
            x_ = torch.exp(x_)
            y_ = torch.exp(y_)

        return super().metrics(x_, y_)

class VideoTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self._y_to_logits = {}
        self._vid_to_logits = {}
        self._vid_to_label = {}

        # TODO needed to extract the first element of y, which includes the video idea; there should be a cleaner pattern to this
        import copy
        loss_fn = copy.deepcopy(self.loss)
        self.loss = lambda x, y: loss_fn(x, y[0])
        self.loss = U.discard_kwargs(self.loss)  # remove extra kwargs
        if hasattr(self, 'loss_val'):
            loss_val_fn = copy.deepcopy(self.loss_val)
            self.loss_val = lambda x, y: loss_val_fn(x, y[0])
            self.loss_val = U.discard_kwargs(self.loss_val)  # remove extra kwargs

    def metrics(self, logits, y, **kwargs):
        labels, vids = y
        return super().metrics(logits, labels, **kwargs)

    def torchmetrics(self, logits, y, prefix):
        """
        logits: (batch, n_classes)
        y = tuple of labels and video ids
        labels: (batch)
        vids: (batch)
        """
        for _logits, _label, _vid in zip(logits, y[0], y[1]):
            _vid = _vid.item()
            # Check that labels are consistent per video id
            assert self._vid_to_label[prefix].get(_vid, _label) == _label
            self._vid_to_label[prefix][_vid] = _label

            self._vid_to_logits[prefix][_vid].append(_logits)

    def _reset_torchmetrics(self, prefix):
        self._vid_to_logits[prefix] = collections.defaultdict(list)
        self._vid_to_label[prefix] = {}

    def get_torchmetrics(self, prefix):
        vid_to_average_logits = {vid: torch.mean(torch.stack(logits, dim=0), dim=0) for vid, logits in self._vid_to_logits[prefix].items()}
        # y is (label, vid) pair
        all_labels = torch.stack(list(self._vid_to_label[prefix].values()), dim=0) # (n_videos)
        all_logits = torch.stack(list(vid_to_average_logits.values()), dim=0) # (n_videos, n_classes)
        m = M.accuracy(all_logits, all_labels)
        return {'aggregate_accuracy': m}


class AdaptiveLMTask(BaseTask):
    def __init__(
        self,
        div_val,
        cutoffs : List[int],
        tie_weights : bool,
        tie_projs : List[bool],
        init_scale=1.0,
        bias_scale=0.0,
        dropemb=0.0,
        dropsoft=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        n_tokens = self.dataset.n_tokens
        d_model = self.model.d_model
        d_output = self.model.d_output

        encoder = AdaptiveEmbedding(
            n_tokens,
            d_model,
            d_model,
            cutoffs=cutoffs,
            div_val=div_val,
            init_scale=init_scale,
            dropout=dropemb,
        )

        if tie_weights:
            assert d_model == d_output
            emb_layers = [i.weight for i in encoder.emb_layers]
        else:
            emb_layers = None

        # Construct decoder/loss
        emb_projs = encoder.emb_projs
        loss = ProjectedAdaptiveLogSoftmax(
            n_tokens, d_output, d_output,
            cutoffs, div_val=div_val,
            tie_projs=tie_projs,
            out_projs=emb_projs,
            out_layers_weights=emb_layers,
            bias_scale=bias_scale,
            dropout=dropsoft,
        )

        self.encoder = encoder
        self.loss = loss


class ImageNetTask(BaseTask):
    """
    Imagenet training uses mixup augmentations, which require a separate loss for train and val,
    which we overide the base task here.

    Not really used anymore.
    """

    def __init__(self, **kwargs):
        import hydra

        super().__init__(
            dataset=kwargs.get("dataset", None),
            model=kwargs.get("model", None),
            loss=kwargs.get("loss", None),  # we still create the base loss here, but will overide below
            metrics=kwargs.get("metrics", None),
            torchmetrics=kwargs.get("torchmetrics", None)
        )

        # if using mixup, overide loss (train) and loss_val, otherwise
        # we have just one loss from the base task above
        if "loss_val" in kwargs and "loss_train" in kwargs:
            self.loss = hydra.utils.instantiate(kwargs.get("loss_train"))
            self.loss_val = hydra.utils.instantiate(kwargs.get('loss_val'))


registry = {
    'base': BaseTask,
    'lm': LMTask,
    'adaptivelm': AdaptiveLMTask,
    'imagenet': ImageNetTask,
    'forecasting': ForecastingTask,
    'video': VideoTask,
}
