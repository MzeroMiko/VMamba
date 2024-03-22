"""Implementation of standard Copying dataset.

Originally used in Arjovsky's Unitary RNN, maybe earlier?
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.utils import distributed


def np_copying_data(L, M, A, batch_shape=()):
    seq = np.random.randint(low=1, high=A-1, size=batch_shape+(M,))
    zeros_x = np.zeros(batch_shape+(L,))
    markers = (A-1) * np.ones(batch_shape+(M,))
    zeros_y = np.zeros(batch_shape+(M+L,))

    x_ = np.concatenate([seq, zeros_x, markers], axis=-1)
    y_ = np.concatenate([zeros_y, seq], axis=-1)
    x = F.one_hot(torch.tensor(x_, dtype=torch.int64), A).float()
    y = torch.tensor(y_, dtype=torch.int64)
    return x, y

def torch_copying_data(L, M, A, variable=False, variable_length=False, batch_shape=(), one_hot=False, reverse=False):
    if variable_length:
        M = int(random.random() * M) + 1
    tokens = torch.randint(low=1, high=A-1, size=batch_shape+(M,))
    if variable:
        total_batch = int(np.prod(batch_shape))
        inds = torch.stack([
            torch.randperm(L+M)[:M]
            for _ in range(total_batch)
            ], 0)
        inds = inds.reshape(batch_shape+(M,))
        inds, _ = inds.sort()
    else:
        inds = torch.arange(M).repeat(batch_shape+(1,))
    zeros_x = torch.zeros(batch_shape+(M+L,), dtype=torch.long)
    zeros_x.scatter_(-1, inds, tokens)
    markers = (A-1) * torch.ones(batch_shape+(M,), dtype=torch.long)

    x_ = torch.cat([zeros_x, markers], dim=-1)
    y_ = torch.cat([tokens], dim=-1)
    if reverse: y_ = y_.flip(-1)
    if one_hot: x = F.one_hot(x_, A).float()
    else: x = x_
    y = y_
    return x, y

def torch_copying_lag_data(L, M, A, batch_shape=()):
    x = torch.randint(low=1, high=A-1, size=batch_shape+(L,))
    y = F.pad(x, (M, 0))[..., :L]
    return x, y

class CopyingTrainDataset(torch.utils.data.Dataset):
    def __init__(self, L, M, A, samples, lag=False, variable=False, variable_length=False, one_hot=False, reverse=False):
        """
        L: number of noise tokens
        M: number of memorization tokens
        A: size of dictionary
        """
        super().__init__()
        self.L = L
        self.M = M
        self.A = A
        self.samples = samples
        self.variable = variable
        self.variable_length = variable_length
        self.one_hot = one_hot
        self.lag = lag
        self.reverse = reverse

    def __getitem__(self, idx):
        assert 0 <= idx < self.samples
        if self.lag:
            x, y = torch_copying_lag_data(self.L, self.M, self.A)
        else:
            x, y = torch_copying_data(self.L, self.M, self.A, variable=self.variable, variable_length=self.variable_length, one_hot=self.one_hot, reverse=self.reverse)
        return x, y

    def __len__(self):
        return self.samples


class CopyingEvalDataset(torch.utils.data.TensorDataset):
    def __init__(self, L, M, A, samples, lag=None, variable=False, variable_length=False, one_hot=False, reverse=False):
        self.L = L
        self.M = M
        self.A = A
        self.samples = samples
        if lag:
            all_x, all_y = torch_copying_lag_data(self.L, self.M, self.A, batch_shape=(self.samples,))
        else:
            all_x, all_y = torch_copying_data(self.L, self.M, self.A, batch_shape=(self.samples,), variable=variable, variable_length=False, one_hot=one_hot, reverse=reverse)
        super().__init__(all_x, all_y)

def copying_static_dataset(L, M, A, variable, samples):
    all_x, all_y = torch_copying_data(L, M, A, variable, batch_shape=(samples,))
    print("Constructing Copying dataset of shape", all_x.shape)
    ds = torch.utils.data.TensorDataset(all_x, all_y)
    return ds
