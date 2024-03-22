"""Audio datasets and utilities."""

import os
from os import listdir
from os.path import join

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from src.dataloaders.base import default_data_path, SequenceDataset, deprecated


def minmax_scale(tensor, range_min=0, range_max=1):
    """
    Min-max scaling to [0, 1].
    """
    min_val = torch.amin(tensor, dim=(1, 2), keepdim=True)
    max_val = torch.amax(tensor, dim=(1, 2), keepdim=True)
    return range_min + (range_max - range_min) * (tensor - min_val) / (max_val - min_val + 1e-6)

def quantize(samples, bits=8, epsilon=0.01):
    """
    Linearly quantize a signal in [0, 1] to a signal in [0, q_levels - 1].
    """
    q_levels = 1 << bits
    samples *= q_levels - epsilon
    samples += epsilon / 2
    return samples.long()

def dequantize(samples, bits=8):
    """
    Dequantize a signal in [0, q_levels - 1].
    """
    q_levels = 1 << bits
    return samples.float() / (q_levels / 2) - 1

def mu_law_encode(audio, bits=8):
    """
    Perform mu-law companding transformation.
    """
    mu = torch.tensor((1 << bits) - 1)

    # Audio must be min-max scaled between -1 and 1
    audio = minmax_scale(audio, range_min=-1, range_max=1)

    # Perform mu-law companding transformation.
    numerator = torch.log1p(mu * torch.abs(audio + 1e-8))
    denominator = torch.log1p(mu)
    encoded = torch.sign(audio) * (numerator / denominator)

    # Shift signal to [0, 1]
    encoded = (encoded + 1) / 2

    # Quantize signal to the specified number of levels.
    return quantize(encoded, bits=bits)

def mu_law_decode(encoded, bits=8):
    """
    Perform inverse mu-law transformation.
    """
    mu = (1 << bits) - 1
    # Invert the quantization
    x = dequantize(encoded, bits=bits)

    # Invert the mu-law transformation
    x = torch.sign(x) * ((1 + mu)**(torch.abs(x)) - 1) / mu

    # Returned values in range [-1, 1]
    return x

def linear_encode(samples, bits=8):
    """
    Perform scaling and linear quantization.
    """
    samples = samples.clone()
    samples = minmax_scale(samples)
    return quantize(samples, bits=bits)

def linear_decode(samples, bits=8):
    """
    Invert the linear quantization.
    """
    return dequantize(samples, bits=bits)

def q_zero(bits=8):
    """
    The quantized level of the 0.0 value.
    """
    return 1 << (bits - 1)


class AbstractAudioDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        bits=8,
        sample_len=None,
        quantization='linear',
        return_type='autoregressive',
        drop_last=True,
        target_sr=None,
        context_len=None,
        pad_len=None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.bits = bits
        self.sample_len = sample_len
        self.quantization = quantization
        self.return_type = return_type
        self.drop_last = drop_last
        self.target_sr = target_sr
        self.zero = q_zero(bits)
        self.context_len = context_len
        self.pad_len = pad_len

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.file_names = NotImplementedError("Must be assigned in setup().")
        self.transforms = {}

        self.setup()
        self.create_quantizer(self.quantization)
        self.create_examples(self.sample_len)


    def setup(self):
        return NotImplementedError("Must assign a list of filepaths to self.file_names.")

    def __getitem__(self, index):
        # Load signal
        if self.sample_len is not None:
            file_name, start_frame, num_frames = self.examples[index]
            seq, sr = torchaudio.load(file_name, frame_offset=start_frame, num_frames=num_frames)
        else:
            seq, sr = torchaudio.load(self.examples[index])

        # Average non-mono signals across channels
        if seq.shape[0] > 1:
            seq = seq.mean(dim=0, keepdim=True)

        # Resample signal if required
        if self.target_sr is not None and sr != self.target_sr:
            if sr not in self.transforms:
                self.transforms[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            seq = self.transforms[sr](seq)

        # Transpose the signal to get (L, 1)
        seq = seq.transpose(0, 1)

        # Unsqueeze to (1, L, 1)
        seq = seq.unsqueeze(0)

        # Quantized signal
        qseq = self.quantizer(seq, self.bits)

        # Squeeze back to (L, 1)
        qseq = qseq.squeeze(0)

        # Return the signal
        if self.return_type == 'autoregressive':
            # Autoregressive training
            # x is [0,  qseq[0], qseq[1], ..., qseq[-2]]
            # y is [qseq[0], qseq[1], ..., qseq[-1]]
            y = qseq
            x = torch.roll(qseq, 1, 0) # Roll the signal 1 step
            x[0] = self.zero # Fill the first element with q_0
            x = x.squeeze(1) # Squeeze to (L, )
            if self.context_len is not None:
                y = y[self.context_len:] # Trim the signal
            if self.pad_len is not None:
                x = torch.cat((torch.zeros(self.pad_len, dtype=self.qtype) + self.zero, x)) # Pad the signal
            return x, y
        elif self.return_type is None:
            return qseq
        else:
            raise NotImplementedError(f'Invalid return type {self.return_type}')

    def __len__(self):
        return len(self.examples)

    def create_examples(self, sample_len: int):
        # Get metadata for all files
        self.metadata = [
            torchaudio.info(file_name) for file_name in self.file_names
        ]

        if sample_len is not None:
            # Reorganize files into a flat list of (file_name, start_frame) pairs
            # so that consecutive items are separated by sample_len
            self.examples = []
            for file_name, metadata in zip(self.file_names, self.metadata):
                # Update the sample_len if resampling to target_sr is required
                # This is because the resampling will change the length of the signal
                # so we need to adjust the sample_len accordingly (e.g. if downsampling
                # the sample_len will need to be increased)
                sample_len_i = sample_len
                if self.target_sr is not None and metadata.sample_rate != self.target_sr:
                    sample_len_i = int(sample_len * metadata.sample_rate / self.target_sr)

                margin = metadata.num_frames % sample_len_i
                for start_frame in range(0, metadata.num_frames - margin, sample_len_i):
                    self.examples.append((file_name, start_frame, sample_len_i))

                if margin > 0 and not self.drop_last:
                    # Last (leftover) example is shorter than sample_len, and equal to the margin
                    # (must be padded in collate_fn)
                    self.examples.append((file_name, metadata.num_frames - margin, margin))
        else:
            self.examples = self.file_names

    def create_quantizer(self, quantization: str):
        if quantization == 'linear':
            self.quantizer = linear_encode
            self.dequantizer = linear_decode
            self.qtype = torch.long
        elif quantization == 'mu-law':
            self.quantizer = mu_law_encode
            self.dequantizer = mu_law_decode
            self.qtype = torch.long
        elif quantization is None:
            self.quantizer = lambda x, bits: x
            self.dequantizer = lambda x, bits: x
            self.qtype = torch.float
        else:
            raise ValueError('Invalid quantization type')

class QuantizedAudioDataset(AbstractAudioDataset):
    """
    Adapted from https://github.com/deepsound-project/samplernn-pytorch/blob/master/dataset.py
    """

    def __init__(
        self,
        path,
        bits=8,
        ratio_min=0,
        ratio_max=1,
        sample_len=None,
        quantization='linear', # [linear, mu-law]
        return_type='autoregressive', # [autoregressive, None]
        drop_last=False,
        target_sr=None,
        context_len=None,
        pad_len=None,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            sample_len=sample_len,
            quantization=quantization,
            return_type=return_type,
            drop_last=drop_last,
            target_sr=target_sr,
            path=path,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            context_len=context_len,
            pad_len=pad_len,
            **kwargs,
        )

    def setup(self):
        from natsort import natsorted
        file_names = natsorted(
            [join(self.path, file_name) for file_name in listdir(self.path)]
        )
        self.file_names = file_names[
            int(self.ratio_min * len(file_names)) : int(self.ratio_max * len(file_names))
        ]

class QuantizedAutoregressiveAudio(SequenceDataset):
    _name_ = 'qautoaudio'

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return 1 << self.bits

    @property
    def l_output(self):
        return self.sample_len

    @property
    def n_tokens(self):
        return 1 << self.bits

    @property
    def init_defaults(self):
        return {
            'path': None,
            'bits': 8,
            'sample_len': None,
            'train_percentage': 0.88,
            'quantization': 'linear',
            'drop_last': False,
            'context_len': None,
            'pad_len': None,
        }

    def setup(self):
        from src.dataloaders.audio import QuantizedAudioDataset
        assert self.path is not None or self.data_dir is not None, "Pass a path to a folder of audio: either `data_dir` for full directory or `path` for relative path."
        if self.data_dir is None:
            self.data_dir = default_data_path / self.path

        self.dataset_train = QuantizedAudioDataset(
            path=self.data_dir,
            bits=self.bits,
            ratio_min=0,
            ratio_max=self.train_percentage,
            sample_len=self.sample_len,
            quantization=self.quantization,
            drop_last=self.drop_last,
            context_len=self.context_len,
            pad_len=self.pad_len,
        )

        self.dataset_val = QuantizedAudioDataset(
            path=self.data_dir,
            bits=self.bits,
            ratio_min=self.train_percentage,
            ratio_max=self.train_percentage + (1 - self.train_percentage) / 2,
            sample_len=self.sample_len,
            quantization=self.quantization,
            drop_last=self.drop_last,
            context_len=self.context_len,
            pad_len=self.pad_len,
        )

        self.dataset_test = QuantizedAudioDataset(
            path=self.data_dir,
            bits=self.bits,
            ratio_min=self.train_percentage + (1 - self.train_percentage) / 2,
            ratio_max=1,
            sample_len=self.sample_len,
            quantization=self.quantization,
            drop_last=self.drop_last,
            context_len=self.context_len,
            pad_len=self.pad_len,
        )

        def collate_fn(batch):
            x, y, *z = zip(*batch)
            assert len(z) == 0
            lengths = torch.tensor([len(e) for e in x])
            max_length = lengths.max()
            if self.pad_len is None:
                pad_length = int(min(2**max_length.log2().ceil(), self.sample_len) - max_length)
            else:
                pad_length = int(min(2**max_length.log2().ceil(), self.sample_len + self.pad_len) - max_length)
            x = nn.utils.rnn.pad_sequence(
                x,
                padding_value=self.dataset_train.zero,
                batch_first=True,
            )
            x = F.pad(x, (0, pad_length), value=self.dataset_train.zero)
            y = nn.utils.rnn.pad_sequence(
                y,
                padding_value=-100, # pad with -100 to ignore these locations in cross-entropy loss
                batch_first=True,
            )
            return x, y, {"lengths": lengths}

        if not self.drop_last:
            self._collate_fn = collate_fn # TODO not tested

class SpeechCommands09(AbstractAudioDataset):

    CLASSES = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]

    CLASS_TO_IDX = dict(zip(CLASSES, range(len(CLASSES))))

    def __init__(
        self,
        path,
        bits=8,
        split='train',
        sample_len=16000,
        quantization='linear', # [linear, mu-law]
        return_type='autoregressive', # [autoregressive, None]
        drop_last=False,
        target_sr=None,
        dequantize=False,
        pad_len=None,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            sample_len=sample_len,
            quantization=quantization,
            return_type=return_type,
            split=split,
            drop_last=drop_last,
            target_sr=target_sr,
            path=path,
            dequantize=dequantize,
            pad_len=pad_len,
            **kwargs,
        )

    def setup(self):
        with open(join(self.path, 'validation_list.txt')) as f:
            validation_files = set([line.rstrip() for line in f.readlines()])

        with open(join(self.path, 'testing_list.txt')) as f:
            test_files = set([line.rstrip() for line in f.readlines()])

        # Get all files in the paths named after CLASSES
        self.file_names = []
        for class_name in self.CLASSES:
            self.file_names += [
                (class_name, file_name)
                for file_name in listdir(join(self.path, class_name))
                if file_name.endswith('.wav')
            ]

        # Keep files based on the split
        if self.split == 'train':
            self.file_names = [
                join(self.path, class_name, file_name)
                for class_name, file_name in self.file_names
                if join(class_name, file_name) not in validation_files
                and join(class_name, file_name) not in test_files
            ]
        elif self.split == 'validation':
            self.file_names = [
                join(self.path, class_name, file_name)
                for class_name, file_name in self.file_names
                if join(class_name, file_name) in validation_files
            ]
        elif self.split == 'test':
            self.file_names = [
                join(self.path, class_name, file_name)
                for class_name, file_name in self.file_names
                if join(class_name, file_name) in test_files
            ]

    def __getitem__(self, index):
        item = super().__getitem__(index)
        x, y, *z = item
        if self.dequantize:
            x = self.dequantizer(x).unsqueeze(1)
        return x, y, *z

class SpeechCommands09Autoregressive(SequenceDataset):
    _name_ = 'sc09'

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return 1 << self.bits

    @property
    def l_output(self):
        return self.sample_len

    @property
    def n_tokens(self):
        return 1 << self.bits

    @property
    def init_defaults(self):
        return {
            'bits': 8,
            'quantization': 'mu-law',
            'dequantize': False,
            'pad_len': None,
        }

    def setup(self):
        from src.dataloaders.audio import SpeechCommands09
        self.data_dir = self.data_dir or default_data_path / self._name_

        self.dataset_train = SpeechCommands09(
            path=self.data_dir,
            bits=self.bits,
            split='train',
            quantization=self.quantization,
            dequantize=self.dequantize,
            pad_len=self.pad_len,
        )

        self.dataset_val = SpeechCommands09(
            path=self.data_dir,
            bits=self.bits,
            split='validation',
            quantization=self.quantization,
            dequantize=self.dequantize,
            pad_len=self.pad_len,
        )

        self.dataset_test = SpeechCommands09(
            path=self.data_dir,
            bits=self.bits,
            split='test',
            quantization=self.quantization,
            dequantize=self.dequantize,
            pad_len=self.pad_len,
        )

        self.sample_len = self.dataset_train.sample_len

    def _collate_fn(self, batch):
        x, y, *z = zip(*batch)
        assert len(z) == 0
        lengths = torch.tensor([len(e) for e in x])
        max_length = lengths.max()
        if self.pad_len is None:
            pad_length = int(min(2**max_length.log2().ceil(), self.sample_len) - max_length)
        else:
            pad_length = 0 # int(self.sample_len + self.pad_len - max_length)
        x = nn.utils.rnn.pad_sequence(
            x,
            padding_value=self.dataset_train.zero if not self.dequantize else 0.,
            batch_first=True,
        )
        x = F.pad(x, (0, pad_length), value=self.dataset_train.zero if not self.dequantize else 0.)
        y = nn.utils.rnn.pad_sequence(
            y,
            padding_value=-100, # pad with -100 to ignore these locations in cross-entropy loss
            batch_first=True,
        )
        y = F.pad(y, (0, 0, 0, pad_length), value=-100) # (batch, length, 1)
        return x, y, {"lengths": lengths}

class MaestroDataset(AbstractAudioDataset):

    YEARS = [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]
    SPLITS = ['train', 'validation', 'test']

    def __init__(
        self,
        path,
        bits=8,
        split='train',
        sample_len=None,
        quantization='linear',
        return_type='autoregressive',
        drop_last=False,
        target_sr=16000,
    ):
        super().__init__(
            bits=bits,
            sample_len=sample_len,
            quantization=quantization,
            return_type=return_type,
            split=split,
            path=path,
            drop_last=drop_last,
            target_sr=target_sr,
        )

    def setup(self):
        import pandas as pd
        from natsort import natsorted

        self.path = str(self.path)

        # Pull out examples in the specified split
        df = pd.read_csv(self.path + '/maestro-v3.0.0.csv')
        df = df[df['split'] == self.split]

        file_names = []
        for filename in df['audio_filename'].values:
            filepath = os.path.join(self.path, filename)
            assert os.path.exists(filepath)
            file_names.append(filepath)
        self.file_names = natsorted(file_names)

class MaestroAutoregressive(SequenceDataset):
    _name_ = 'maestro'

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return 1 << self.bits

    @property
    def l_output(self):
        return self.sample_len

    @property
    def n_tokens(self):
        return 1 << self.bits

    @property
    def init_defaults(self):
        return {
            'bits': 8,
            'sample_len': None,
            'quantization': 'mu-law',
        }

    def setup(self):
        from src.dataloaders.audio import MaestroDataset
        self.data_dir = self.data_dir or default_data_path / self._name_ / 'maestro-v3.0.0'

        self.dataset_train = MaestroDataset(
            path=self.data_dir,
            bits=self.bits,
            split='train',
            sample_len=self.sample_len,
            quantization=self.quantization,
        )

        self.dataset_val = MaestroDataset(
            path=self.data_dir,
            bits=self.bits,
            split='validation',
            sample_len=self.sample_len,
            quantization=self.quantization,
        )

        self.dataset_test = MaestroDataset(
            path=self.data_dir,
            bits=self.bits,
            split='test',
            sample_len=self.sample_len,
            quantization=self.quantization,
        )

    def _collate_fn(self, batch):
        x, y, *z = zip(*batch)
        assert len(z) == 0
        lengths = torch.tensor([len(e) for e in x])
        max_length = lengths.max()
        pad_length = int(min(max(1024, 2**max_length.log2().ceil()), self.sample_len) - max_length)
        x = nn.utils.rnn.pad_sequence(
            x,
            padding_value=self.dataset_train.zero,
            batch_first=True,
        )
        x = F.pad(x, (0, pad_length), value=self.dataset_train.zero)
        y = nn.utils.rnn.pad_sequence(
            y,
            padding_value=self.dataset_train.zero,
            batch_first=True,
        )
        return x, y, {"lengths": lengths}

class LJSpeech(QuantizedAudioDataset):

    def __init__(
        self,
        path,
        bits=8,
        ratio_min=0,
        ratio_max=1,
        sample_len=None,
        quantization='linear', # [linear, mu-law]
        return_type='autoregressive', # [autoregressive, None]
        drop_last=False,
        target_sr=None,
        use_text=False,
    ):
        super().__init__(
            bits=bits,
            sample_len=sample_len,
            quantization=quantization,
            return_type=return_type,
            drop_last=drop_last,
            target_sr=target_sr,
            path=path,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            use_text=use_text,
        )

    def setup(self):
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        super().setup()

        self.vocab_size = None
        if self.use_text:
            self.transcripts = {}
            with open(str(self.path.parents[0] / 'metadata.csv'), 'r') as f:
                for line in f:
                    index, raw_transcript, normalized_transcript = line.rstrip('\n').split("|")
                    self.transcripts[index] = normalized_transcript
            # df = pd.read_csv(self.path.parents[0] / 'metadata.csv', sep="|", header=None)
            # self.transcripts = dict(zip(df[0], df[2])) # use normalized transcripts

            self.tok_transcripts = {}
            self.vocab = set()
            for file_name in self.file_names:
                # Very simple tokenization, character by character
                # Capitalization is ignored for simplicity
                file_name = file_name.split('/')[-1].split('.')[0]
                self.tok_transcripts[file_name] = list(self.transcripts[file_name].lower())
                self.vocab.update(self.tok_transcripts[file_name])

            # Fit a label encoder mapping characters to numbers
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(list(self.vocab))
            # add a token for padding, no additional token for UNK (our dev/test set contain no unseen characters)
            self.vocab_size = len(self.vocab) + 1

            # Finalize the tokenized transcripts
            for file_name in self.file_names:
                file_name = file_name.split('/')[-1].split('.')[0]
                self.tok_transcripts[file_name] = torch.tensor(self.label_encoder.transform(self.tok_transcripts[file_name]))


    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.use_text:
            file_name, _, _ = self.examples[index]
            tok_transcript = self.tok_transcripts[file_name.split('/')[-1].split('.')[0]]
            return *item, tok_transcript
        return item

class LJSpeechAutoregressive(SequenceDataset):
    _name_ = 'ljspeech'

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return 1 << self.bits

    @property
    def l_output(self):
        return self.sample_len

    @property
    def n_tokens(self):
        return 1 << self.bits

    @property
    def init_defaults(self):
        return {
            'bits': 8,
            'sample_len': None,
            'quantization': 'mu-law',
            'train_percentage': 0.88,
            'use_text': False,
        }

    def setup(self):
        from src.dataloaders.audio import LJSpeech
        self.data_dir = self.data_dir or default_data_path / self._name_ / 'LJSpeech-1.1' / 'wavs'

        self.dataset_train = LJSpeech(
            path=self.data_dir,
            bits=self.bits,
            ratio_min=0,
            ratio_max=self.train_percentage,
            sample_len=self.sample_len,
            quantization=self.quantization,
            target_sr=16000,
            use_text=self.use_text,
        )

        self.dataset_val = LJSpeech(
            path=self.data_dir,
            bits=self.bits,
            ratio_min=self.train_percentage,
            ratio_max=self.train_percentage + (1 - self.train_percentage) / 2,
            sample_len=self.sample_len,
            quantization=self.quantization,
            target_sr=16000,
            use_text=self.use_text,
        )

        self.dataset_test = LJSpeech(
            path=self.data_dir,
            bits=self.bits,
            ratio_min=self.train_percentage + (1 - self.train_percentage) / 2,
            ratio_max=1,
            sample_len=self.sample_len,
            quantization=self.quantization,
            target_sr=16000,
            use_text=self.use_text,
        )

        self.vocab_size = self.dataset_train.vocab_size

    def _collate_fn(self, batch):
        x, y, *z = zip(*batch)

        if self.use_text:
            tokens = z[0]
            text_lengths = torch.tensor([len(e) for e in tokens])
            tokens = nn.utils.rnn.pad_sequence(
                tokens,
                padding_value=self.vocab_size - 1,
                batch_first=True,
            )
        else:
            assert len(z) == 0
        lengths = torch.tensor([len(e) for e in x])
        max_length = lengths.max()
        pad_length = int(min(2**max_length.log2().ceil(), self.sample_len) - max_length)
        x = nn.utils.rnn.pad_sequence(
            x,
            padding_value=self.dataset_train.zero,
            batch_first=True,
        )
        x = F.pad(x, (0, pad_length), value=self.dataset_train.zero)
        y = nn.utils.rnn.pad_sequence(
            y,
            padding_value=-100, # pad with -100 to ignore these locations in cross-entropy loss
            batch_first=True,
        )
        if self.use_text:
            return x, y, {"lengths": lengths, "tokens": tokens, "text_lengths": text_lengths}
        else:
            return x, y, {"lengths": lengths}

class _SpeechCommands09Classification(SpeechCommands09):

    def __init__(
        self,
        path,
        bits=8,
        split='train',
        sample_len=16000,
        quantization='linear', # [linear, mu-law]
        drop_last=False,
        target_sr=None,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            sample_len=sample_len,
            quantization=quantization,
            return_type=None,
            split=split,
            drop_last=drop_last,
            target_sr=target_sr,
            path=path,
            **kwargs,
        )

    def __getitem__(self, index):
        x = super().__getitem__(index)
        x = mu_law_decode(x)
        y = torch.tensor(self.CLASS_TO_IDX[self.file_names[index].split("/")[-2]])
        return x, y

class SpeechCommands09Classification(SequenceDataset):
    _name_ = 'sc09cls'

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return 10

    @property
    def l_output(self):
        return 0

    @property
    def n_tokens(self):
        return 1 << self.bits

    @property
    def init_defaults(self):
        return {
            'bits': 8,
            'quantization': 'mu-law',
        }

    def setup(self):
        from src.dataloaders.audio import _SpeechCommands09Classification
        self.data_dir = self.data_dir or default_data_path / 'sc09'

        self.dataset_train = _SpeechCommands09Classification(
            path=self.data_dir,
            bits=self.bits,
            split='train',
            quantization=self.quantization,
        )

        self.dataset_val = _SpeechCommands09Classification(
            path=self.data_dir,
            bits=self.bits,
            split='validation',
            quantization=self.quantization,
        )

        self.dataset_test = _SpeechCommands09Classification(
            path=self.data_dir,
            bits=self.bits,
            split='test',
            quantization=self.quantization,
        )

        self.sample_len = self.dataset_train.sample_len

    def collate_fn(self, batch):
        x, y, *z = zip(*batch)
        assert len(z) == 0
        lengths = torch.tensor([len(e) for e in x])
        max_length = lengths.max()
        pad_length = int(min(2**max_length.log2().ceil(), self.sample_len) - max_length)
        x = nn.utils.rnn.pad_sequence(
            x,
            padding_value=self.dataset_train.zero,
            batch_first=True,
        )
        x = F.pad(x, (0, pad_length), value=0.)#self.dataset_train.zero)
        y = torch.tensor(y)
        return x, y, {"lengths": lengths}

@deprecated
class SpeechCommandsGeneration(SequenceDataset):
    _name_ = "scg"

    init_defaults = {
        "mfcc": False,
        "dropped_rate": 0.0,
        "length": 16000,
        "all_classes": False,
        "discrete_input": False,
    }

    @property
    def n_tokens(self):
        return 256 if self.discrete_input else None

    def init(self):
        if self.mfcc:
            self.d_input = 20
            self.L = 161
        else:
            self.d_input = 1
            self.L = self.length

        if self.dropped_rate > 0.0:
            self.d_input += 1

        self.d_output = 256
        self.l_output = self.length

    def setup(self):
        from src.dataloaders.datasets.sc import _SpeechCommandsGeneration

        # TODO refactor with data_dir argument
        self.dataset_train = _SpeechCommandsGeneration(
            partition="train",
            length=self.length,  # self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=default_data_path,
            all_classes=self.all_classes,
            discrete_input=self.discrete_input,
        )

        self.dataset_val = _SpeechCommandsGeneration(
            partition="val",
            length=self.length,  # self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=default_data_path,
            all_classes=self.all_classes,
            discrete_input=self.discrete_input,
        )

        self.dataset_test = _SpeechCommandsGeneration(
            partition="test",
            length=self.length,  # self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=default_data_path,
            all_classes=self.all_classes,
            discrete_input=self.discrete_input,
        )

    @classmethod
    def _return_callback(cls, return_value, *args, **kwargs):
        x, y, *z = return_value
        return x, y.long(), *z

@deprecated
class Music(SequenceDataset):
    _name_ = "music"

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return 256

    @property
    def l_output(self):
        return self.sample_rate * self.sample_len

    @property
    def n_tokens(self):
        return 256 if self.discrete_input else None

    @property
    def init_defaults(self):
        return {
            "sample_len": 1,
            "sample_rate": 16000,
            "train_percentage": 0.88,
            "discrete_input": False,
        }

    def init(self):
        return

    def setup(self):
        from src.dataloaders.music import _Music

        self.music_class = _Music(
            path=default_data_path,
            sample_len=self.sample_len,  # In seconds
            sample_rate=self.sample_rate,
            train_percentage=self.train_percentage,  # Use settings from SampleRNN paper
            discrete_input=self.discrete_input,
        )

        self.dataset_train = self.music_class.get_data("train")
        self.dataset_test = self.music_class.get_data("test")
        self.dataset_val = self.music_class.get_data("val")

    @classmethod
    def _return_callback(cls, return_value, *args, **kwargs):
        x, y, *z = return_value
        return x, y.long(), *z
