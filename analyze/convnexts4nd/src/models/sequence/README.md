This folder contains implementations of a general sequence modeling framework.

```
base.py      Base SequenceModule interface
backbones/   Modular DNN backbones with flexible configuration
attention/   Implementations of attention variants
convs/       Implementations of basic (local) convolutions
kernels/     Modules for wide conv kernels that S4 and related works use
rnns/        Implementations of RNN models
modules/     Other sequence-to-sequence modules
```

# Modular Sequence Model Interface

This README provides a basic overview of the sequence model source code.
It is recommended to see the [config README](/configs/model/README.md) for running experiments with these models.


## SequenceModule
The SequenceModule class ([base.py](base.py)) is the abstract interface that all sequence models adhere to.
In this codebase, sequence models are defined as a sequence-to-sequence map of shape `(batch size, sequence length, model dimension)` to `(batch size, sequence length, output dimension)`.

The SequenceModule comes with other methods such as `step` which is meant for autoregressive settings, and logic to carry optional hidden states (for stateful models such as RNNs or S4).

To add a new model to this codebase, subclass `SequenceModule` and implement the required methods.

## SequenceModel
The `SequenceModel` class ([model.py](model.py)) is the main backbone with configurable options for residual function, normalization placement, etc.

SequenceModel accepts a black box config for a layer. Compatible layers are SequenceModules (i.e. composable sequence transformations) found under this `sequence/` folder.

## Layers

### S4 (and other convolution kernels)

The end-to-end S4 model consists of a vanilla convolution block [modules/s4block.py](modules/s4block.py) that accepts any convolution kernel.
These kernels are defined under [kernels/](kernels/), including S4 variants ([kernels/ssm.py](./kernels/ssm.py)) and other generic convolution kernels ([kernels/kernel.py](./kernels/kernel.py)).

### Attention and Convolutions

Variants of attention (standard MHA as well as [Linear Attention](https://arxiv.org/abs/2006.16236) and [Performer](https://arxiv.org/abs/2009.14794)) are under [attention/](attention/).
Simple Conv1D and Conv2D wrappers are under [convs/](convs/).

### RNNs

This codebase also contains a modular implementation of many RNN cells.
These include HiPPO-RNN cells from the original [HiPPO paper](https://arxiv.org/abs/2008.07669).

Some examples include `model=rnn/hippo-legs` and `model=rnn/hippo-legt` for HiPPO variants from the original [paper](https://arxiv.org/abs/2008.07669), or `model=rnn/gru` for a GRU reimplementation, etc.

An exception is `model=lstm` to use the PyTorch LSTM.

Example command (reproducing the Permuted MNIST number from the HiPPO paper, which was SotA at the time):
```
python train.py pipeline=mnist model=rnn/hippo-legs model.cell_args.hidden_size=512 train.epochs=50 train.batch_size=100 train.lr=0.001
```

### DNN blocks and other modules

[modules/](modules/) has other modules which all adhere to the SequenceModule (sequence-to-sequence transformation) interface, including the FFN block of Transformers, different DNN blocks such as the [Mega block](https://arxiv.org/abs/2209.10655) which combines an S4 variant with attention variant, a generic pooling layer, and so on.
