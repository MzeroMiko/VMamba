# Overview

Basic datasets including MNIST, CIFAR, and Speech Commands will auto-download. Source code for these datamodules are in [basic.py](basic.py).

By default, data is downloaded to `./data/`  by default, where `.` is the top level directory of this repository (e.g. 'state-spaces').

- [Data Preparation](#data-preparation) - Instructions for downloading other datasets
- [Adding a Dataset](#adding-a-dataset-wip) - Basic instructions for adding new datasets

## Advanced Usage

After downloading and preparing data, the paths can be configured in several ways.

1. Suppose that it is desired to download all data to a different folder, for example a different disk.
The data path can be configured by setting the environment variable `DATA_PATH`, which defaults to `./data`.

2. For fine-grained control over the path of a particular dataset, set `dataset.data_dir` in the config. For example, if the LRA ListOps files are located in `/home/lra/listops-1000/` instead of the default `./data/listops/`,
pass in `+dataset.data_dir=/home/lra/listops-1000` on the command line or modify the config file directly.

3. As a simple workaround, softlinks can be set, e.g. `ln -s /home/lra/listops-1000 ./data/listops`


# Data Preparation

Datasets that must be manually downloaded include [LRA](#long-range-arena-lra), [WikiText-103](#wikitext-103), [BIDMC](#bidmc), and [other audio datasets](#other-audio) used in SaShiMi.

By default, these should go under `$DATA_PATH/`, which defaults to `./data`.  For the remainder of this README, these are used interchangeably.

## Long Range Arena (LRA)

LRA can be downloaded from the [GitHub page](https://github.com/google-research/long-range-arena).
These datasets should be organized as follows:
```
$DATA_PATH/
  pathfinder/
    pathfinder32/
    pathfinder64/
    pathfinder128/
    pathfinder256/
  aan/
  listops/
```
The other two datasets in the suite ("Image" i.e. grayscale sequential CIFAR-10; "Text" i.e. char-level IMDB sentiment classification) are both auto-downloaded.

The following sequence of commands prepares the LRA datasets in the default data path:
```
cd data
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
tar xvf lra_release.gz
mv lra_release/lra_release/listops-1000 listops
mv lra_release/lra_release/tsv_data aan
mkdir pathfinder
mv lra_release/lra_release/pathfinder* pathfinder/
rm -r lra_release
```

## Speech Commands (SC)

The full SC dataset is auto-downloaded into `./data/SpeechCommands/`.
Specific subsets such as the SC10 subset can be toggled in the config or command line.

For the SC09 audio generation dataset, copy the digit subclasses of the `./data/SpeechCommands` folder into `data/sc09/{zero,one,two,three,four,five,six,seven,eight,nine}`. Also copy the `./data/SpeechCommands/{validation_list,test_list}.txt` files.

## WikiText-103

The WikiText-103 language modeling dataset can be downloaded by the `getdata.sh` script from the [Transformer-XL codebase](https://github.com/kimiyoung/transformer-xl).
By default, the datamodule looks for it under `$DATA_PATH/wt103`.

```
cd {repo}/data
wget https://raw.githubusercontent.com/kimiyoung/transformer-xl/master/getdata.sh
bash getdata.sh
mv data/wikitext-103 wt103
```

A trained model checkpoint can be found [here](https://huggingface.co/krandiash/sashimi-release/tree/main/checkpoints). (Note that this uses a vanilla isotropic S4 model and is only located in the SaShiMi release for convenience.)

## BIDMC

See [prepare/bidmc/README.md](prepare/bidmc/README.md)

## Informer Forecasting Datasets

The ETTH, ETTM, Weather, and ECL experiments originally from the [Informer]() paper can be downloaded as [informer.zip](https://drive.google.com/file/d/1XqpxE6cthIxKYviSmR703yU45vdQ1oHT/view?usp=sharing) and extracted inside `./data`.



## Other Audio

Instructions for other audio datasets used by the SaShiMi paper, including Beethoven and YoutubeMix,
can be found in the [SaShiMi README](/models/sashimi/).

# Adding a Dataset [WIP]
Datasets generally consist of two components.

1. The first is the `torch.utils.data.Dataset` class which defines the raw data, or (data, target) pairs.

2. The second is a [SequenceDataset](/src/dataloaders/base.py) class, which defines how to set up the dataset as well as the dataloaders. This class is very similar to PyTorch Lightning's `LightningDataModule` and satisfies an interface described below.

Datasets are sometimes defined in the [datasets/](./datasets/) subfolder, while Datamodules are all defined in the top-level files in this folder and imported by [\_\_init\_\_.py](./__init__.py).

Basic examples of datamodules are provided [here](./basic.py).

Some help for adding a custom audio dataset was provided in Issue https://github.com/HazyResearch/state-spaces/issues/23

## SequenceDataset [WIP]

TODO:
- Add documentation for adding a new dataset
- Restructure folder so that each dataset is in its own file
- Use Hydra to instantiate datamodules

<!--
## Basic interface
- `_name_`



Examples of pipelines

- Audio/images (e.g. SC) have a Resolution argument that needs to be passed through
- IMDB


- TODO what's a case when decoder needs to return something else to pass into the loss function?

- PassthroughSequential
  - used to stack encoders and decoders
  - used in AdaptiveLM?


- LMTask: embedding, scaling, (optional) positional embedding
- forecasting: consume time features
- conditional generation: ClassEmbedding (extra data is the class)

- original dataloader arguments are also passed to the decoder, in case they need to use them (e.g. sequence lengths). Note that sometimes extra arguments may be passed in which will also get passed through the pipeline

- is there a case where we might want an encoder to return additional arguments? e.g. encode a length into a mask which is passed into every layer tensor and additional arguments?
-->
