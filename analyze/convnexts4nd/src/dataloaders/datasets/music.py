"""RNN Vocal Generation Model.

Blizzard, Music, and Huckleberry Finn data feeders.
"""

import numpy as np
#import scikits.audiolab

import random
import time
import os
import glob

import torch
import sklearn
from scipy.io import wavfile

def normalize01(data):
     """To range [0., 1.]"""
     data -= np.min(data)
     data /= np.max(data)
     return data

def mu_law_encode(audio, bits=8):
    """
    Perform mu-law companding transformation.
    """
    mu = torch.tensor(2**bits - 1)

    # Audio must be min-max scaled between -1 and 1
    audio = 2 * minmax_scale(audio) - 1

    # Perform mu-law companding transformation.
    numerator = torch.log1p(mu * torch.abs(audio + 1e-8))
    denominator = torch.log1p(mu)
    encoded = torch.sign(audio) * (numerator / denominator)

    # Quantize signal to the specified number of levels.
    return ((encoded + 1) / 2 * mu + 0.5).long()

def mu_law_decode(encoded, bits=8):
    """
    Perform inverse mu-law transformation.
    """
    mu = 2**bits - 1
    # Invert the quantization
    x = (encoded.float() / mu) * 2 - 1

    # Invert the mu-law transformation
    x = torch.sign(x) * ((1 + mu)**(torch.abs(x)) - 1) / mu
    return x

def minmax_scale(tensor):
    min_val = torch.amin(tensor, dim=(1, 2), keepdim=True)
    max_val = torch.amax(tensor, dim=(1, 2), keepdim=True)
    return (tensor - min_val) / (max_val - min_val + 1e-6)

EPSILON = 1e-2

def linear_quantize(samples, q_levels):
    samples = samples.clone()
    # samples -= samples.min(dim=-2)[0].unsqueeze(1).expand_as(samples)
    # samples /= samples.max(dim=-2)[0].unsqueeze(1).expand_as(samples)
    samples = minmax_scale(samples)
    samples *= q_levels - EPSILON
    samples += EPSILON / 2
    return samples.long()

def linear_dequantize(samples, q_levels):
    return samples.float() / (q_levels / 2) - 1

def q_zero(q_levels):
    return q_levels // 2

ITEM_LIST = [
    "BeethovenPianoSonataNo.1",
    "BeethovenPianoSonataNo.2",
    "BeethovenPianoSonataNo.3",
    "BeethovenPianoSonataNo.4",
    "BeethovenPianoSonataNo.5",
    "BeethovenPianoSonataNo.6",
    "BeethovenPianoSonataNo.7",
    "BeethovenPianoSonataNo.8",
    "BeethovenPianoSonataNo.9",
    "BeethovenPianoSonataNo.10",
    "BeethovenPianoSonataNo.11",
    "BeethovenPianoSonataNo.12",
    "BeethovenPianoSonata13",
    "BeethovenPianoSonataNo.14moonlight",
    "BeethovenPianoSonata15",
    "BeethovenPianoSonata16",
    "BeethovenPianoSonata17",
    "BeethovenPianoSonataNo.18",
    "BeethovenPianoSonataNo.19",
    "BeethovenPianoSonataNo.20",
    "BeethovenPianoSonataNo.21Waldstein",
    "BeethovenPianoSonata22",
    "BeethovenPianoSonataNo.23",
    "BeethovenPianoSonataNo.24",
    "BeethovenPianoSonataNo.25",
    "BeethovenPianoSonataNo.26",
    "BeethovenPianoSonataNo.27",
    "BeethovenPianoSonataNo.28",
    "BeethovenPianoSonataNo.29",
    "BeethovenPianoSonataNo.30",
    "BeethovenPianoSonataNo.31",
    "BeethovenPianoSonataNo.32",
]

def download_all_data(path):
    print('Downloading data to ' + path)
    if not os.path.exists(path):
        os.system('mkdir ' + path)
    for item in ITEM_LIST:
        os.system("wget -r -H -nc -nH --cut-dir=1 -A .ogg -R *_vbr.mp3 -e robots=off -P " + path + " -l1 'http://archive.org/download/" + item + "'")
        os.system("mv " + os.path.join(path, item, '*.ogg') + " " + path)
        os.system("rm -rf " + os.path.join(path, item))
    for f in os.listdir(path):
        filepath = os.path.join(path, f)
        os.system("ffmpeg -y -i " + filepath + " -ar 16000 -ac 1 " + filepath[:-4] + ".wav")
        os.system("rm " + filepath)
    print('Data download done')

class _Music():
    def __init__(
            self,
            path,
            sample_len = 1,  # in seconds
            sample_rate = 16000,
            train_percentage = 0.9,
            discrete_input=False,
            samplernn_proc=True,
        ):
        self.sample_len = sample_len
        self.sample_rate = sample_rate
        self.discrete_input = discrete_input
        self.samplernn_proc = samplernn_proc

        self.music_data_path = os.path.join(path, 'music_data')
        if not os.path.exists(self.music_data_path):
            download_all_data(self.music_data_path)

        self.all_data = self.get_all_data()
        self.tensor = self.build_slices(self.all_data)
        self.train, self.val, self.test = self.split_data(self.tensor, train_percentage)
        self.train_X, self.val_X, self.test_X, self.train_y, self.val_y, self.test_y = self.make_x_y(self.train, self.val, self.test)


    def get_all_data(self):
        from librosa.core import load
        # TODO: There are going to be boundary errors here!
        all_data = np.array([])
        for f in os.listdir(self.music_data_path):
            # sr, data = wavfile.read(os.path.join(self.music_data_path, f))
            data, _ = load(os.path.join(self.music_data_path, f), sr=None, mono=True)
            # assert(sr == self.sample_rate)
            all_data = np.append(all_data, data)

        # # if not self.samplernn_proc:
        # # Convert all data to range [-1, 1]
        # all_data = all_data.astype('float64')
        # all_data = normalize01(all_data)
        # all_data = 2. * all_data - 1.

        return all_data

    def build_slices(self, data):
        num_samples_per_slice = self.sample_rate * self.sample_len

        truncated_len = len(data) - len(data) % num_samples_per_slice

        return torch.tensor(data[:truncated_len].reshape(-1, num_samples_per_slice), dtype=torch.float32)

        # tensor = torch.zeros([len(data) // num_samples_per_slice, num_samples_per_slice], dtype=torch.float32)
        # for i in range(len(data) // num_samples_per_slice):
        #     tensor[i] = torch.tensor(data[i * num_samples_per_slice : (i + 1) * num_samples_per_slice])
        # return tensor

    def split_data(self, tensor, train_percentage):
        train, test = sklearn.model_selection.train_test_split(
                tensor,
                train_size=train_percentage,
                random_state=0,
                shuffle=True
        )
        val, test = sklearn.model_selection.train_test_split(
                test,
                train_size=0.5,
                random_state=0,
                shuffle=True
        )
        train = torch.swapaxes(train.unsqueeze(1).squeeze(-1), 1, 2)
        val = torch.swapaxes(val.unsqueeze(1).squeeze(-1), 1, 2)
        test = torch.swapaxes(test.unsqueeze(1).squeeze(-1), 1, 2)
        return train, val, test

    def make_x_y(self, train, val, test):

        if not self.samplernn_proc:
            train_y, val_y, test_y = mu_law_encode(train), mu_law_encode(val), mu_law_encode(test)
            if not self.discrete_input:
                train_X, val_X, test_X = torch.roll(mu_law_decode(train_y), 1, 1), torch.roll(mu_law_decode(val_y), 1, 1), torch.roll(mu_law_decode(test_y), 1, 1)
                train_X[:, 0, :], val_X[:, 0, :], test_X[:, 0, :] = 0, 0, 0
            else:
                train_X, val_X, test_X = torch.roll(train_y, 1, 1), torch.roll(val_y, 1, 1), torch.roll(test_y, 1, 1)
                train_X[:, 0, :], val_X[:, 0, :], test_X[:, 0, :] = 128, 128, 128
        else:
            train_y, val_y, test_y = linear_quantize(train, 256), linear_quantize(val, 256), linear_quantize(test, 256)
            # train_y, val_y, test_y = mu_law_encode(train), mu_law_encode(val), mu_law_encode(test)
            if not self.discrete_input:
                raise NotImplementedError
            else:
                train_X, val_X, test_X = torch.roll(train_y, 1, 1), torch.roll(val_y, 1, 1), torch.roll(test_y, 1, 1)
                train_X[:, 0, :], val_X[:, 0, :], test_X[:, 0, :] = 128, 128, 128

        return train_X, val_X, test_X, train_y, val_y, test_y

    def get_data(self, partition):
        if partition == 'train':
            return MusicTensorDataset(self.train_X, self.train_y)
        elif partition == 'val':
            return MusicTensorDataset(self.val_X, self.val_y)
        elif partition == 'test':
            return MusicTensorDataset(self.test_X, self.test_y)

class MusicTensorDataset(torch.utils.data.TensorDataset):

    def __getitem__(self, index):
        data = self.tensors[0][index]
        target = self.tensors[1][index]
        if data.dtype == torch.float32:
            return data, target
        else:
            return data.squeeze(-1), target
        # Rejection sampling to remove "bad samples" that are essentially constant audio
        # if data.dtype == torch.float32:
        #     if torch.std(data[1:]) < 1e-5:
        #         return self.__getitem__(np.random.randint(0, len(self.tensors[0])))
        #     return data, target
        # else:
        #     if (data[1:] - data[1]).abs().sum() < 1e-5:
        #         return self.__getitem__(np.random.randint(0, len(self.tensors[0])))
        #     return data.squeeze(-1), target

