"""Implementation of standard Adding dataset.

Originally used in Arjovsky's Unitary RNN, maybe earlier?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_adding_data(L, batch_shape=()):
    assert L >= 2
    mid = L//2
    idx0 = torch.randint(low=0, high=mid, size=batch_shape)
    idx1 = torch.randint(low=0, high=L-mid, size=batch_shape)

    idx = torch.cat((F.one_hot(idx0, mid), F.one_hot(idx1, L-mid)), dim=-1).float() # (batch_shape, L)
    unif = torch.empty(batch_shape+(L,))
    unif.uniform_(0., 1.)

    x = torch.stack((unif, idx), dim=-1) # (batch_shape, L, 2)
    y = torch.sum(unif*idx, dim=-1, keepdim=True) # (batch_shape, 1)

    return x, y

def adding_static_dataset(L, samples):
    all_x, all_y = torch_adding_data(L, batch_shape=(samples,))
    print("Constructing Adding dataset of shape", all_x.shape)
    ds = torch.utils.data.TensorDataset(all_x, all_y)
    return ds
