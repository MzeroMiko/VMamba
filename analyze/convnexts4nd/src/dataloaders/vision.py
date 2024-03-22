"""Miscellaneous vision datasets."""

import os

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from src.dataloaders.base import default_data_path, SequenceDataset

class CIFAR100(SequenceDataset):
    _name_ = "cifar100"
    d_output = 100
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "permute": None,
            "grayscale": False,
            "tokenize": False,  # if grayscale, tokenize into discrete byte inputs
            "augment": False,
            "cutout": False,
            "random_erasing": False,
            "val_split": 0.1,
            "seed": 42,  # For validation split
        }

    @property
    def d_input(self):
        if self.grayscale:
            if self.tokenize:
                return 256
            else:
                return 1
        else:
            assert not self.tokenize
            return 3

    def setup(self):
        if self.grayscale:
            preprocessors = [
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
            ]
            permutations_list = [
                torchvision.transforms.Lambda(
                    lambda x: x.view(1, 1024).t()
                )  # (L, d_input)
            ]

            if self.tokenize:
                preprocessors.append(
                    torchvision.transforms.Lambda(lambda x: (x * 255).long())
                )
                permutations_list.append(Rearrange("l 1 -> l"))
            else:
                preprocessors.append(
                    torchvision.transforms.Normalize(
                        mean=122.6 / 255.0, std=61.0 / 255.0
                    )
                )
        else:
            preprocessors = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
                ),
            ]
            permutations_list = [
                torchvision.transforms.Lambda(
                    Rearrange("z h w -> (h w) z", z=3, h=32, w=32)
                )  # (L, d_input)
            ]

        # Permutations and reshaping
        if self.permute == "br":
            permutation = permutations.bitreversal_permutation(1024)
            print("bit reversal", permutation)
            permutations_list.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "snake":
            permutation = permutations.snake_permutation(32, 32)
            print("snake", permutation)
            permutations_list.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "hilbert":
            permutation = permutations.hilbert_permutation(32)
            print("hilbert", permutation)
            permutations_list.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "transpose":
            permutation = permutations.transpose_permutation(32, 32)
            transform = torchvision.transforms.Lambda(
                lambda x: torch.cat([x, x[permutation]], dim=-1)
            )
            permutations_list.append(transform)
        elif self.permute == "2d":  # h, w, c
            permutation = torchvision.transforms.Lambda(
                    Rearrange("(h w) c -> h w c", h=32, w=32)
                )
            permutations_list.append(permutation)
        elif self.permute == "2d_transpose":  # c, h, w
            permutation = torchvision.transforms.Lambda(
                    Rearrange("(h w) c -> c h w", h=32, w=32)
                )
            permutations_list.append(permutation)

        # Augmentation
        if self.augment:
            augmentations = [
                torchvision.transforms.RandomCrop(
                    32, padding=4, padding_mode="symmetric"
                ),
                torchvision.transforms.RandomHorizontalFlip(),
            ]

            post_augmentations = []
            if self.cutout:
                post_augmentations.append(Cutout(1, 16))
                pass
            if self.random_erasing:
                # augmentations.append(RandomErasing())
                pass
        else:
            augmentations, post_augmentations = [], []
        torchvision.transforms_train = (
            augmentations + preprocessors + post_augmentations + permutations_list
        )
        torchvision.transforms_eval = preprocessors + permutations_list

        transform_train = torchvision.transforms.Compose(torchvision.transforms_train)
        transform_eval = torchvision.transforms.Compose(torchvision.transforms_eval)
        self.dataset_train = torchvision.datasets.CIFAR100(
            f"{default_data_path}/{self._name_}",
            train=True,
            download=True,
            transform=transform_train,
        )
        self.dataset_test = torchvision.datasets.CIFAR100(
            f"{default_data_path}/{self._name_}", train=False, transform=transform_eval
        )
        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"


class CIFAR10C(SequenceDataset):
    """
    Make sure to specify a corruption using e.g. `dataset.corruption=gaussian_blur`.
    Corruption options are: ['brightness', 'contrast', 'defocus_blur',
            'elastic_transform', 'fog', 'frost', 'gaussian_blur', 'gaussian_noise',
            'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur',
            'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter',
            'speckle_noise', 'zoom_blur']

    A full example of a command using this dataset:
    `python -m train wandb=null experiment=s4-cifar dataset=cifar-c +train.validate_at_start=true dataset.corruption=gaussian_blur`

    Note that the metric people use for CIFAR-C is mean corruption error (mCE), normalized by
    the accuracy AlexNet gets on the dataset. You can use this spreadsheet to calculate mCE:
    https://docs.google.com/spreadsheets/d/1RwqofJPHhtdRPG-dDO7wPp-aGn-AmwmU5-rpvTzrMHw
    """
    _name_ = "cifar-c"
    d_output = 10
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "corruption": None,
        }

    @property
    def d_input(self):
        return 3

    def setup(self):
        from src.dataloaders.datasets.cifarc import _CIFAR10C
        self.data_dir = self.data_dir or default_data_path / "CIFAR-10-C"

        # make sure self.corruptions was specified and is a valid choice
        assert self.corruption != None, "You must specify a corruption. Options are: " + \
            str(sorted([p.stem for p in self.data_dir.glob("*.npy") if not p.stem == 'labels']))
        assert os.path.isfile(os.path.join(self.data_dir,f"{self.corruption}.npy")), \
            f"Corruption '{self.corruption}' does not exist. Options are: " + \
                str(sorted([p.stem for p in self.data_dir.glob("*.npy") if not p.stem == 'labels']))

        preprocessors = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            ),
        ]
        permutations_list = [
            torchvision.transforms.Lambda(
                Rearrange("z h w -> (h w) z", z=3, h=32, w=32)
            )
        ]

        transform_eval = torchvision.transforms.Compose(preprocessors + permutations_list)

        x = np.load(os.path.join(self.data_dir,f"{self.corruption}.npy"))
        y = np.load(os.path.join(self.data_dir,"labels.npy"))

        self.dataset_test = _CIFAR10C(x, y, transform_eval)

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"


class CIFAR10Generation(SequenceDataset):
    """TODO there should be a way to combine this with main CIFAR class. the issue is making sure the torchvision.transforms are applied to output in the same way."""

    _name_ = "cifargen"

    @property
    def init_defaults(self):
        return {
            "transpose": False,
            "tokenize": True,
            "mixture": 0,
            "val_split": 0.02,
            "seed": 42,
        }

    @property
    def d_input(self):
        if not self.tokenize:
            return 1  # Returns None otherwise

    @property
    def d_output(self):
        return 256 if self.mixture == 0 else 3 * self.mixture

    @property
    def n_tokens(self):
        if self.tokenize:
            return 3 * 256 + 1

    @property
    def n_classes(self):  # TODO not used?
        return 10

    @property
    def permute(self):
        if self.transpose:  # R R ... G G ... B B ...
            return lambda x: rearrange(x, "... h w c -> ... (c h w) 1")
        else:  # R G B R G B ...
            return lambda x: rearrange(x, "... h w c -> ... (h w c) 1")

    @property
    def transforms0(self):
        """Transforms applied before permutation"""
        if self.tokenize:
            return torchvision.transforms.Lambda(
                lambda x: x + 1 + torch.arange(3) * 256
            )
        else:
            # return torchvision.transforms.Normalize(mean=127.5, std=127.5)
            return torchvision.transforms.Lambda(lambda x: (x.float() - 127.5) / 127.5)

    @property
    def transforms1(self):
        """Transforms applied after permutation"""
        if self.tokenize:
            return torchvision.transforms.Lambda(lambda x: x.squeeze(-1))
        else:
            return torchvision.transforms.Compose([])

    def setup(self):
        transforms = [
            torchvision.transforms.ToTensor(),  # (B, C, H, W)
            Rearrange("c h w -> h w c"),  # (B, H, W, C)
            torchvision.transforms.Lambda(
                lambda x: (x * 255).long()
            ),  # Convert back to ints
        ]
        transform = torchvision.transforms.Compose(transforms)

        self.dataset_train = torchvision.datasets.CIFAR10(
            f"{default_data_path}/cifar",
            train=True,
            download=True,
            transform=transform,
        )
        self.dataset_test = torchvision.datasets.CIFAR10(
            f"{default_data_path}/cifar", train=False, transform=transform
        )
        self.split_train_val(self.val_split)

        def collate_batch(batch):
            """batch: list of (x, y) pairs"""
            inputs, labels = zip(*batch)
            x = torch.stack(inputs, dim=0)
            z = torch.LongTensor(labels)
            y = self.permute(x)
            x = self.transforms0(x)
            x = self.permute(x)
            x = F.pad(x[:, :-1, :], (0, 0, 1, 0))
            x = self.transforms1(x)
            return x, y, z

        self.collate_fn = collate_batch

    def __str__(self):  # TODO not updated
        return f"{self._name_}"


class CIFAR10GenerationFactored(CIFAR10Generation):
    """Version of CIFAR-10 Density Estimation that keeps the sequence of length 1024 and factors the distribution over the 3 channels"""

    _name_ = "cifargenf"
    l_output = 1024  # Leaving this out or setting to None also works, to indicate that the entire length dimension is kept

    @property
    def init_defaults(self):
        return {
            "mixture": 0,
            "val_split": 0.02,
            "seed": 42,
        }

    @property
    def d_input(self):
        return 3

    @property
    def d_output(self):
        return 3 * 256 if self.mixture == 0 else 10 * self.mixture

    @property
    def permute(self):
        return lambda x: rearrange(x, "... h w c -> ... (h w) c")

    @property
    def transforms0(self):
        return torchvision.transforms.Lambda(lambda x: (x.float() - 127.5) / 127.5)
        # return torchvision.transforms.Normalize(mean=0.5, std=0.5)

    @property
    def transforms1(self):
        return torchvision.transforms.Compose([])


class HMDB51(SequenceDataset):
    # TODO(KG): refactor this dataset with new SequenceDataset structure

    _name_ = "hmdb51"
    d_input = 3
    d_output = 51
    l_output = 0

    init_defaults = {
        "split_dir": "test_train_splits",  # path to splits
        "video_dir": "videos",  # path to videos
        "clip_duration": 2,  # Duration of sampled clip for each video, just the upper bound
        "num_frames": 16,  # frames per clip
        "frame_size": 112,  # square shape of image to use
        "use_ddp": False,  # using a distributed sampler / not
        "num_gpus": 1,
        "split_id": 1,  # 1, 2, or 3
        "val_split": 0.1,  # split train into val also
        "augment": "default",  # which type of augment to use, "default" | "randaug" | "augmix"
        # "num_rand_augments": 3,  # num of random augmentations to use
        # "use_augmix": False
    }

    def split_train_val(self, val_split):
        """
        Child class needs to handle getting length of dataset differently.
        """
        train_len = int(self.dataset_train.num_videos * (1.0 - val_split))
        self.dataset_train, self.dataset_val = random_split(
            self.dataset_train,
            (train_len, self.dataset_train.num_videos - train_len),
            generator=torch.Generator().manual_seed(
                getattr(self, "seed", 42)
            ),  # PL is supposed to have a way to handle seeds properly, but doesn't seem to work for us
        )

    def find_classes(self, directory):
        """Finds the class folders in a dataset.
        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def setup(self):

        # for video datasets
        import pytorch_lightning
        import pytorchvideo.data
        import torch.utils.data
        from torch.utils.data import DistributedSampler, RandomSampler

        self.pytorchvideo = pytorchvideo.data
        self.RandomSampler = RandomSampler
        self.DistributedSampler = DistributedSampler

        from pytorchvideo.transforms import (ApplyTransformToKey, AugMix,
                                             Normalize, Permute, RandAugment,
                                             RandomShortSideScale, RemoveKey,
                                             ShortSideScale,
                                             UniformTemporalSubsample)
        from torchvision.transforms import (CenterCrop, Compose, Lambda,
                                            RandomCrop, RandomHorizontalFlip,
                                            Resize)

        self.split_path = self.data_dir or default_data_path / self._name_
        self.split_path = os.path.join(self.split_path, self.split_dir)
        self.video_path = self.data_dir or default_data_path / self._name_
        self.video_path = os.path.join(self.video_path, self.video_dir)

        # # sampler = RandomSampler  # hardcode, ddp handled by PTL
        # sampler = DistributedSampler if self.num_gpus > 1 else RandomSampler
        # print("sampler chosen!", sampler)

        # means = (0.43216, 0.394666, 0.37645)
        # stds = (0.22803, 0.22145, 0.216989)
        means = (110.2, 100.64, 96.0)
        stds = (58.14765, 56.46975, 55.332195)

        train_transform_list = []

        train_transform_list += [UniformTemporalSubsample(self.num_frames),
                            Lambda(lambda x: x / 255.0),
                            Normalize(means, stds)]

        if self.augment == "randaug": aug_paras = self.randaug
        elif self.augment == "augmix": aug_paras = self.augmix
        else: aug_paras = None
        self.train_transform = pytorchvideo.transforms.create_video_transform(
            mode="train",
            video_key="video",
            num_samples=self.num_frames,
            convert_to_float=False,
            video_mean=means,
            video_std=stds,
            min_size=256, # for ShortSideScale
            crop_size=self.frame_size,
            aug_type=self.augment,
            aug_paras=aug_paras,
        )
        self.test_transform = pytorchvideo.transforms.create_video_transform(
            mode="val",
            video_key="video",
            num_samples=self.num_frames,
            convert_to_float=False,
            video_mean=means,
            video_std=stds,
            min_size=256, # for ShortSideScale
            crop_size=self.frame_size,
            aug_type=self.augment,
            aug_paras=aug_paras,
        )
        # get list of classes, and class_to_idx, to convert class str to int val
        self.classes, self.class_to_idx = self.find_classes(self.video_path)

        # @staticmethod
        def collate_batch(batch, resolution=1):

            videos, str_labels, video_idxs = zip(
                *[
                    (data["video"], data["label"], data["video_index"])
                    for data in batch
                ]
            )

            # need to convert label string to int, and then to tensors
            int_labels = [torch.tensor(self.class_to_idx[label]) for label in str_labels]
            video_idx_labels = [torch.tensor(label) for label in video_idxs]  # just convert to tensor

            xs = torch.stack(videos)  # shape = [b, c, t, h, w]
            ys = torch.stack(int_labels)
            video_idxs = torch.stack(video_idx_labels)

            return xs, (ys, video_idxs)

        self.collate_fn = collate_batch

    def train_dataloader(self, **kwargs):
        """Need to overide so that we don't pass the shuffle=True parameter"""

        sampler = self.DistributedSampler if self.num_gpus > 1 else self.RandomSampler

        self.dataset_train = self.pytorchvideo.Hmdb51(
            data_path=self.split_path,
            video_path_prefix=self.video_path,
            clip_sampler=self.pytorchvideo.make_clip_sampler("random", self.clip_duration),
            decode_audio=False,
            split_id=self.split_id,
            split_type="train",
            transform=self.train_transform,
            video_sampler=sampler
        )

        return torch.utils.data.DataLoader(
            self.dataset_train,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def val_dataloader(self, **kwargs):

        kwargs['drop_last'] = False
        sampler = partial(self.DistributedSampler, drop_last=kwargs['drop_last']) if self.num_gpus > 1 else self.RandomSampler

        self.dataset_val = self.pytorchvideo.Hmdb51(
            data_path=self.split_path,
            video_path_prefix=self.video_path,
            clip_sampler=self.pytorchvideo.make_clip_sampler("uniform", self.clip_duration),
            decode_audio=False,
            split_id=self.split_id,
            split_type="test",
            transform=self.test_transform,
            video_sampler=sampler
        )

        return torch.utils.data.DataLoader(
            self.dataset_val,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def test_dataloader(self, **kwargs):

        kwargs['drop_last'] = False
        sampler = partial(self.DistributedSampler, drop_last=kwargs['drop_last']) if self.num_gpus > 1 else self.RandomSampler

        self.dataset_test = self.pytorchvideo.Hmdb51(
            data_path=self.split_path,
            video_path_prefix=self.video_path,
            clip_sampler=self.pytorchvideo.make_clip_sampler("uniform", self.clip_duration),
            decode_audio=False,
            split_id=self.split_id,
            split_type="test",
            transform=self.test_transform,
            video_sampler=sampler
        )

        return torch.utils.data.DataLoader(
            self.dataset_test,
            collate_fn=self.collate_fn,
            **kwargs,
        )

class ImageNet(SequenceDataset):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/
        Sample-of-Images-from-the-ImageNet-Dataset-used-in-the-ILSVRC-Challenge.png
        :width: 400
        :alt: Imagenet
    Specs:
        - 1000 classes
        - Each image is (3 x varies x varies) (here we default to 3 x 224 x 224)
    Imagenet train, val and test dataloaders.
    The train set is the imagenet train.
    The val split is taken from train if a val_split % is provided, or will be the same as test otherwise
    The test set is the official imagenet validation set.

    """

    _name_ = "imagenet"
    d_input = 3
    d_output = 1000
    l_output = 0

    init_defaults = {
        "data_dir": None,
        "cache_dir": None,
        "image_size": 224,
        "val_split": None,  # currently not implemented
        "train_transforms": None,
        "val_transforms": None,
        "test_transforms": None,
        "mixup": None,  # augmentation
        "num_aug_repeats": 0,
        "num_gpus": 1,
        "shuffle": True,  # for train
        "loader_fft": False,
    }

    @property
    def num_classes(self) -> int:
        """
        Return:
            1000
        """
        return 1000

    def _verify_splits(self, data_dir: str, split: str) -> None:
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(
                f"a {split} Imagenet split was not found in {data_dir},"
                f" make sure the folder contains a subfolder named {split}"
            )

    def prepare_data(self) -> None:
        """This method already assumes you have imagenet2012 downloaded. It validates the data using the meta.bin.
        .. warning:: Please download imagenet on your own first.
        """
        if not self.use_archive_dataset:
            self._verify_splits(self.data_dir, "train")
            self._verify_splits(self.data_dir, "val")
        else:
            if not self.data_dir.is_file():
                raise FileNotFoundError(f"""Archive file {str(self.data_dir)} not found.""")

    def setup(self, stage=None):
        """Creates train, val, and test dataset."""

        from typing import Any, Callable, List, Optional, Union

        import hydra  # for mixup
        from pl_bolts.transforms.dataset_normalizations import \
            imagenet_normalization
        from torch.utils.data import Dataset
        from torch.utils.data.dataloader import default_collate
        from torchvision.datasets import ImageFolder

        # for access in other methods
        self.imagenet_normalization = imagenet_normalization
        self.default_collate = default_collate
        self.hydra = hydra
        self.ImageFolder = ImageFolder

        if self.mixup is not None:
            self.mixup_fn = hydra.utils.instantiate(self.mixup)
        else:
            self.mixup_fn = None

        self.dir_path = self.data_dir or default_data_path / self._name_

        if stage == "fit" or stage is None:
            self.set_phase([self.image_size])

        #     train_transforms = (self.train_transform() if self.train_transforms is None
        #                         else hydra.utils.instantiate(self.train_transforms))
        #     val_transforms = (self.val_transform() if self.val_transforms is None
        #                       else hydra.utils.instantiate(self.val_transforms))

        #     self.dataset_train = ImageFolder(self.dir_path / 'val',  # modded
        #                                         transform=train_transforms)

        #     if self.val_split > 0.:
        #         # this will create the val split
        #         self.split_train_val(self.val_split)
        #     # will use the test split as val by default
        #     else:
        #         self.dataset_val = ImageFolder(self.dir_path / 'val', transform=val_transforms)

        #     # modded, override (for debugging)
        #     self.dataset_train = self.dataset_val

        if stage == "test" or stage is None:
            test_transforms = (self.val_transform() if self.test_transforms is None
                               else hydra.utils.instantiate(self.test_transforms))

            self.dataset_test = ImageFolder(os.path.join(self.dir_path, 'val'), transform=test_transforms)

            # # modded, override (for debugging)
            # self.dataset_test = self.dataset_val

    def set_phase(self, stage_params=[224], val_upsample=False, test_upsample=False):
        """
        For progresive learning.
        Will modify train transform parameters during training, just image size for now,
        and create a new train dataset, which the train_dataloader will load every
        n epochs (in config).

        Later, will be possible to change magnitude of RandAug here too, and mixup alpha

        stage_params: list, list of values to change.  single [image_size] for now
        """

        img_size = int(stage_params[0])

        # self.train_transforms["input_size"] = img_size

        if val_upsample:
            self.val_transforms["input_size"] = img_size

        train_transforms = (self.train_transform() if self.train_transforms is None
                            else self.hydra.utils.instantiate(self.train_transforms))
        val_transforms = (self.val_transform() if self.val_transforms is None
                            else self.hydra.utils.instantiate(self.val_transforms))

        if self.loader_fft:
            train_transforms = torchvision.transforms.Compose(
                train_transforms.transforms + [
                    torchvision.transforms.Lambda(lambda x: torch.fft.rfftn(x, s=tuple([2*l for l in x.shape[1:]])))
                ]
            )
            val_transforms = torchvision.transforms.Compose(
                val_transforms.transforms + [
                    torchvision.transforms.Lambda(lambda x: torch.fft.rfftn(x, s=tuple([2*l for l in x.shape[1:]])))
                ]
            )

        self.dataset_train = self.ImageFolder(self.dir_path / 'train',
                                            transform=train_transforms)

        if self.val_split > 0.:
            # this will create the val split
            self.split_train_val(self.val_split)
        # will use the test split as val by default
        else:
            self.dataset_val = self.ImageFolder(self.dir_path / 'val', transform=val_transforms)

        # # modded, override (for debugging)
        # self.dataset_train = self.dataset_val

        # not sure if normally you upsample test also
        if test_upsample:
            self.test_transforms["input_size"] = img_size
            test_transforms = (self.val_transform() if self.test_transforms is None
                                else self.hydra.utils.instantiate(self.test_transforms))
            self.dataset_test = self.ImageFolder(os.path.join(self.dir_path, 'val'), transform=test_transforms)
            ## modded, override (for debugging)
            # self.dataset_test = self.dataset_val

        # could modify mixup by reinstantiating self.mixup_fn (later maybe)

    def train_transform(self):
        """The standard imagenet transforms.
        .. code-block:: python
            transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """
        preprocessing = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(self.image_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                self.imagenet_normalization(),
            ]
        )

        return preprocessing

    def val_transform(self):
        """The standard imagenet transforms for validation.
        .. code-block:: python
            transforms.Compose([
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """

        preprocessing = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.image_size + 32),
                torchvision.transforms.CenterCrop(self.image_size),
                torchvision.transforms.ToTensor(),
                self.imagenet_normalization(),
            ]
        )
        return preprocessing

    # def train_dataloader(self, train_resolution, eval_resolutions, **kwargs):
    #     """ The train dataloader """
    #     return (self._data_loader(self.dataset_train, shuffle=True, mixup=self.mixup_fn, **kwargs))

    def train_dataloader(self, **kwargs):
        """ The train dataloader """
        if self.num_aug_repeats == 0 or self.num_gpus == 1:
            shuffle = self.shuffle
            sampler = None
        else:
            shuffle = False
            from timm.data.distributed_sampler import RepeatAugSampler
            sampler = RepeatAugSampler(self.dataset_train, num_repeats=self.num_aug_repeats)

        # calculate resolution
        resolution = self.image_size / self.train_transforms['input_size']  # usually 1.0

        return (self._data_loader(self.dataset_train, shuffle=shuffle, mixup=self.mixup_fn, sampler=sampler, resolution=resolution, **kwargs))

    def val_dataloader(self, **kwargs):    
        """ The val dataloader """
        kwargs['drop_last'] = False

        # update batch_size for eval if provided
        batch_size = kwargs.get("batch_size_eval", None) or kwargs.get("batch_size")
        kwargs["batch_size"] = batch_size

        # calculate resolution
        resolution = self.image_size / self.val_transforms['input_size']  # usually 1.0 or 0.583

        return (self._data_loader(self.dataset_val, resolution=resolution, **kwargs))

    def test_dataloader(self, **kwargs):    
        """ The test dataloader """
        kwargs['drop_last'] = False

        # update batch_size for test if provided
        batch_size = kwargs.get("batch_size_test", None) or kwargs.get("batch_size_eval", None) or kwargs.get("batch_size")
        kwargs["batch_size"] = batch_size

        # calculate resolution
        resolution = self.image_size / self.test_transforms.get("input_size", self.val_transforms['input_size'])

        return (self._data_loader(self.dataset_test, resolution=resolution, **kwargs))

    def _data_loader(self, dataset, resolution, shuffle=False, mixup=None, sampler=None, **kwargs):
        # collate_fn = (lambda batch: mixup(*self.default_collate(batch))) if mixup is not None else self.default_collate
        collate_fn = (lambda batch: mixup(*self.collate_with_resolution(batch, resolution))) if mixup is not None else lambda batch: self.collate_with_resolution(batch, resolution)

        # hacked - can't pass this this arg to dataloader, but used to update the batch_size val / test
        kwargs.pop('batch_size_eval', None)
        kwargs.pop('batch_size_test', None)

        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs,
        )

    def collate_with_resolution(self, batch, resolution):
        stuff = self.default_collate(batch)
        return *stuff, {"resolution": resolution}

    # def _data_loader(self, dataset, mixup=None, **kwargs):
    #     collate_fn = (lambda batch: mixup(*self.default_collate(batch))) if mixup is not None else self.default_collate
    #     return torch.utils.data.DataLoader(
    #         dataset, collate_fn=collate_fn, **kwargs
    #     )

class ImageNetA(ImageNet):
    _name_ = 'imagenet-a'

    init_defaults = {
        'transforms': None,
    }

    def setup(self):
        from pl_bolts.transforms.dataset_normalizations import \
            imagenet_normalization
        from torch.utils.data.dataloader import default_collate
        from torchvision.datasets import ImageFolder
        self.imagenet_normalization = imagenet_normalization
        self.default_collate = default_collate
        self.ImageFolder = ImageFolder

        self.dir_path = self.data_dir or default_data_path / self._name_

        # self.transforms["input_size"] = 224
        transforms = (
            self.val_transform() if self.transforms is None
            else self.hydra.utils.instantiate(self.transforms)
        )

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = self.ImageFolder(self.dir_path, transform=transforms)

class ImageNetR(ImageNetA):
    _name_ = 'imagenet-r'

class ImageNetC(ImageNet):
    _name_ = 'imagenet-c'

    init_defaults = {
        'transforms': None,
    }

    def setup(self):
        from pl_bolts.transforms.dataset_normalizations import \
            imagenet_normalization
        from torch.utils.data.dataloader import default_collate
        from torchvision.datasets import ImageFolder
        self.imagenet_normalization = imagenet_normalization
        self.default_collate = default_collate
        self.ImageFolder = ImageFolder
        self.dir_path = self.data_dir or default_data_path / self._name_

        # self.transforms["input_size"] = 224
        transforms = (
            self.val_transform() if self.transforms is None
            else self.hydra.utils.instantiate(self.transforms)
        )

        variants = [os.listdir(self.dir_path)][0]
        subvariants = {variant: os.listdir(os.path.join(self.dir_path, variant)) for variant in variants}

        self.dataset_test = {
            f'{variant + "/" + subvariant}': self.ImageFolder(
                os.path.join(os.path.join(self.dir_path, variant), subvariant),
                transform=transforms,
            )
            for variant in variants
            for subvariant in subvariants[variant]
        }

        self.dataset_train = None
        self.dataset_val = None
        # self.dataset_test = self.ImageFolder(self.dir_path, transform=transforms)

    def val_dataloader(self, **kwargs):
        """Using the same dataloader as test, a hack for zero shot eval without training"""
        kwargs['drop_last'] = False
        kwargs["batch_size"] = kwargs.get("batch_size_eval", None) or kwargs.get("batch_size")
        return {
            name: self._data_loader(dataset, resolution=1, **kwargs)
            for name, dataset in self.dataset_test.items()
        }

    def test_dataloader(self, **kwargs):
        kwargs['drop_last'] = False
        kwargs["batch_size"] = kwargs.get("batch_size_eval", None) or kwargs.get("batch_size")
        return {
            name: self._data_loader(dataset, resolution=1, **kwargs)
            for name, dataset in self.dataset_test.items()
        }

class ImageNetP(ImageNet):
    _name_ = 'imagenet-p'

    init_defaults = {
        'transforms': None,
    }

    def setup(self):
        from pl_bolts.transforms.dataset_normalizations import \
            imagenet_normalization
        from src.dataloaders.utils.video_loader import VideoFolder
        from torch.utils.data.dataloader import default_collate
        self.imagenet_normalization = imagenet_normalization
        self.default_collate = default_collate
        self.VideoFolder = VideoFolder
        self.dir_path = self.data_dir or default_data_path / self._name_

        # self.transforms["input_size"] = 224
        transforms = (
            self.val_transform() if self.transforms is None
            else self.hydra.utils.instantiate(self.transforms)
        )

        variants = os.listdir(self.dir_path)
        # subvariants = {variant: os.listdir(os.path.join(self.dir_path, variant)) for variant in variants}

        self.dataset_test = {
            f'{variant}': self.VideoFolder(
                os.path.join(self.dir_path, variant),
                transform=transforms,
            )
            for variant in variants
            # for subvariant in subvariants[variant]
        }

        self.dataset_train = None
        self.dataset_val = None
        # self.dataset_test = self.ImageFolder(self.dir_path, transform=transforms)

    def val_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        """Using the same dataloader as test, a hack for zero shot eval without training"""
        kwargs['drop_last'] = False
        kwargs["batch_size"] = kwargs.get("batch_size_eval", None) or kwargs.get("batch_size")
        return {
            name: self._data_loader(dataset, **kwargs)
            for name, dataset in self.dataset_test.items()
        }

    def test_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        kwargs['drop_last'] = False
        kwargs["batch_size"] = kwargs.get("batch_size_eval", None) or kwargs.get("batch_size")
        return {
            name: self._data_loader(dataset, **kwargs)
            for name, dataset in self.dataset_test.items()
        }
