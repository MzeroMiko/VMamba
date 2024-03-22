This repository provides a modular and flexible implementation of general deep sequence models.

```
baselines/   Ported baseline models
functional/  Mathematical utilities
hippo/       Utilities for defining HiPPO operators
nn/          Standalone neural network components (nn.Module)
s4/          Standalone S4 modules
sequence/    Modular sequence model interface
```

<!-- ## HiPPO -->
<!---->
<!-- HiPPO is the mathematical framework upon which the papers HiPPO, LSSL, and S4 are built on. -->
<!-- HiPPO operators are defined in [hippo/hippo.py](hippo/hippo.py). -->
<!-- Function reconstruction experiments and visualizations are presented in [hippo/visualizations.py](hippo/visualizations.py). -->

## S4

In v3, a standalone implementation of S4 could be found inside `s4/`. It has been moved to [/models/s4/](/models/s4/).
The fully tested S4 implementation is inside [sequence/](sequence/).

## Modular Sequence Model Interface

A general deep sequence model framework can be found in [sequence/](sequence/).
All models and experiments that this repository official supports used this framework.
See [sequence/README.md](sequence/) for more information.

## Baselines
Other sequence models are easily incorporated into this repository,
and several other baselines have been ported.
These include CNNs such as [CKConv](https://arxiv.org/abs/2102.02611) and continuous-time/RNN models such as [UnICORNN](https://arxiv.org/abs/2103.05487) and [LipschitzRNN](https://arxiv.org/abs/2006.12070).

Models and datasets can be flexibly interchanged.
Examples:
```
python -m train pipeline=cifar model=ckconv
python -m train pipeline=mnist model=lipschitzrnn
```

The distinction between baselines in `baselines/` and models in `sequence/` is that
the baselines do not necessarily subscribe to the modular `SequenceModule` interface,
and are usually monolithic end-to-end models adapted from other codebases.
