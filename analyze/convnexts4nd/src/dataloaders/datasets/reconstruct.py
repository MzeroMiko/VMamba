import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dataloaders.utils.signal import whitesignal


class ReconstructTrainDataset(torch.utils.data.Dataset):
    def __init__(self, samples, l_seq=1024, l_mem=1024, dt=1e-3, freq=1.0, seed=0):
        """
        """
        super().__init__()
        self.L = l_seq
        self.l_mem = l_mem
        self.dt = dt
        self.freq = freq
        self.samples = samples

    def __getitem__(self, idx):
        assert 0 <= idx < self.samples
        x = torch.FloatTensor(whitesignal(self.L*self.dt, self.dt, self.freq))
        x = x.unsqueeze(-1)
        y = x[-self.l_mem:, 0]
        return x, y

    def __len__(self):
        return self.samples


class ReconstructEvalDataset(torch.utils.data.TensorDataset):
    def __init__(self, samples, l_seq=1024, l_mem=1024, dt=1e-3, freq=1.0, seed=0):
        self.L = l_seq
        self.l_mem = l_mem
        self.dt = dt
        self.freq = freq
        self.samples = samples

        X = []
        X = torch.FloatTensor(whitesignal(self.L*self.dt, self.dt, self.freq, batch_shape=(self.samples,)))
        X = X[..., None]
        Y = X[:, -self.l_mem:, 0]

        super().__init__(X, Y)
