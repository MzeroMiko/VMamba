"""Synthetic datasets."""

import numpy as np
import torch
import torchvision
from einops.layers.torch import Rearrange
from src.utils import permutations

from src.dataloaders.base import SequenceDataset


class Copying(SequenceDataset):
    _name_ = "copying"

    @property
    def init_defaults(self):
        return {
            "l_noise": 100,  # number of padding tokens
            "l_memorize": 10,  # number of tokens to memorize
            "n_tokens": 10,  # alphabet size
            "lag": False,
            "variable": False,  # Randomly distribute memorization tokens throughout sequence instead of frontloading them
            "variable_length": False,  # Randomize number of tokens to memorize
            "one_hot": False,
            "reverse": False,
            "static": False, # Use a static dataset of size n_train, otherwise always use random data with n_train per epoch
            "n_train": 10000,
            "n_eval": 1000,
        }

    @property
    def d_input(self):
        return self.n_tokens

    @property
    def d_output(self):
        return self.n_tokens

    @property
    def l_output(self):
        return self.l_noise if self.lag else self.l_memorize

    def setup(self):
        from .datasets.copying import CopyingEvalDataset, CopyingTrainDataset

        if self.static: train_cls = CopyingEvalDataset
        else: train_cls = CopyingTrainDataset

        self.dataset_train = train_cls(
            self.l_noise,
            self.l_memorize,
            self.n_tokens,
            samples=self.n_train,
            lag=self.lag,
            variable=self.variable,
            one_hot=self.one_hot,
            reverse=self.reverse,
        )
        self.dataset_val = CopyingEvalDataset(
            self.l_noise,
            self.l_memorize,
            self.n_tokens,
            samples=self.n_eval,
            lag=self.lag,
            variable=self.variable,
            one_hot=self.one_hot,
            reverse=self.reverse,
        )
        self.dataset_test = None


    def __str__(self):
        return f"{self._name_}{self.l_noise}{'v' if self.variable else ''}"


class Adding(SequenceDataset):
    _name_ = "adding"
    d_input = 2
    d_output = 1
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "l_max": 1000,
            "n_samples": 50000,
            "val_split": 0.1,
        }

    def setup(self):
        from .datasets.adding import adding_static_dataset

        self.dataset_train = adding_static_dataset(self.l_max, self.n_samples)
        self.dataset_test = None
        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{self._name_}{self.l_max}"


class Reconstruct(SequenceDataset):
    _name_ = "reconstruct"

    @property
    def init_defaults(self):
        return {
            "l_seq": 1024, # length of total sequence
            "l_mem": 512,  # length to reconstruct
            "dt": 0.001,
            "freq": 1.0,
            "seed": 0,
            "static": False, # Use a static dataset of size n_train, otherwise always use random data with n_train per epoch
            "n_train": 10000,
            "n_eval": 1000,
        }

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return self.l_mem

    @property
    def l_output(self):
        return 0

    def setup(self):
        from .datasets.reconstruct import ReconstructEvalDataset, ReconstructTrainDataset

        if self.static: train_cls = ReconstructEvalDataset
        else: train_cls = ReconstructTrainDataset

        self.dataset_train = train_cls(
            samples=self.n_train,
            l_seq=self.l_seq,
            l_mem=self.l_mem,
            dt=self.dt,
            freq=self.freq,
            seed=self.seed,
        )
        self.dataset_val = ReconstructEvalDataset(
            samples=self.n_eval,
            l_seq=self.l_seq,
            l_mem=self.l_mem,
            dt=self.dt,
            freq=self.freq,
            seed=self.seed,
        )
        self.dataset_test = None

    def __str__(self):
        raise NotImplementedError


class Delay(SequenceDataset):
    _name_ = "delay"

    @property
    def init_defaults(self):
        return {
            "l_seq": 1024, # length of total sequence
            "n_lag": 1,  # length to reconstruct
            "l_lag": None,  # length to reconstruct
            "dt": 0.001,
            "freq": 100.0,
            "static": False, # Use a static dataset of size n_train, otherwise always use random data with n_train per epoch
            "n_train": 10000,
            "n_eval": 1000,
        }

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        # NOTE: To reproduce numbers from HTTYH paper, set this equal to 4. There was a bug in the implementation at the time
        return self.n_lag

    @property
    def l_output(self):
        return self.l_seq

    def setup(self):
        from .datasets.delay import DelayEvalDataset, DelayTrainDataset

        if self.static: train_cls = DelayEvalDataset
        else: train_cls = DelayTrainDataset

        self.dataset_train = train_cls(
            samples=self.n_train,
            l_seq=self.l_seq,
            n_lag=self.n_lag,
            l_lag=self.l_lag,
            dt=self.dt,
            freq=self.freq,
        )
        self.dataset_val = DelayEvalDataset(
            samples=self.n_eval,
            l_seq=self.l_seq,
            n_lag=self.n_lag,
            l_lag=self.l_lag,
            dt=self.dt,
            freq=self.freq,
        )
        self.dataset_test = None


    def __str__(self):
        return f"{self._name_}{self.l_noise}{'v' if self.variable else ''}"

class MackeyGlass(SequenceDataset):
    _name_ = "mackey"

    @property
    def init_defaults(self):
        return {
            "l_seq": 5000, # length of total sequence
            "l_predict": 15,  # length to reconstruct
            "tau": 17,  # Delay of MG system
            "washout": 100,
            "delta_t": 10,
            "n_train": 1024,
            "n_eval": 64,
        }

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return 1

    @property
    def l_output(self):
        return self.l_seq

    def setup(self):
        from .datasets.mackey import mackey_glass

        train_X, train_Y = mackey_glass(
            n_samples=self.n_train,
            l_seq=self.l_seq,
            l_predict=self.l_predict,
            tau=self.tau,
            washout=self.washout,
            delta_t=self.delta_t,
        )
        train_X, train_Y = torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
        val_X, val_Y = mackey_glass(
            n_samples=self.n_eval,
            l_seq=self.l_seq,
            l_predict=self.l_predict,
            tau=self.tau,
            washout=self.washout,
            delta_t=self.delta_t,
        )
        val_X, val_Y = torch.FloatTensor(val_X), torch.FloatTensor(val_Y)
        self.dataset_train = torch.utils.data.TensorDataset(train_X, train_Y)
        self.dataset_val = torch.utils.data.TensorDataset(val_X, val_Y)
        self.dataset_test = None


    def __str__(self):
        return f"{self._name_}"
