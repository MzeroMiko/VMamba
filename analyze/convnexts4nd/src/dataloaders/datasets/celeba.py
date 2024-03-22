"""Implementation of Celeb-A dataset."""

from functools import partial
import torch
import os
import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets import VisionDataset
try:
    import gdown
    DOWNLOAD = True
except ImportError:
    DOWNLOAD = False
import numpy as np

class _CelebA(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("1cNIac61PSA_LqDFYFUeyaQYekYPc75NH", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
            self,
            root: str,
            task: str = None,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        import pandas
        super(_CelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
            "hq": None,
        }
        split_ = split_map[split]

        if split == 'hq':
            fn = partial(os.path.join, self.root)
        else:
            fn = partial(os.path.join, self.root, self.base_folder)

        splits = pandas.read_csv(fn("list_eval_partition.csv"), header=0, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.csv"), header=0, index_col=0)
        mask = slice(None) if split_ is None else (splits['partition'] == split_)

        if split == 'hq':
            filenames = os.listdir(fn('train')) + os.listdir(fn('val'))
            self.filename = [fn('train', f) for f in os.listdir(fn('train'))] + [fn('val', f) for f in os.listdir(fn('val'))]
            self.attr = torch.as_tensor(attr.loc[filenames].values)
        else:
            self.filename = splits[mask].index.values
            self.attr = torch.as_tensor(attr[mask].values)

        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

        self.task = task
        if task:
            self.task_idx = int(np.where(np.array(self.attr_names) == task)[0])

    def download(self) -> None:
        import zipfile
        if not DOWNLOAD:
            raise ImportError("Must install gdown.")

        if os.path.exists(os.path.join(self.root, self.base_folder, 'img_align_celeba')):
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            gdown.download(f'https://drive.google.com/uc?id={file_id}', os.path.join(self.root, self.base_folder, filename), quiet=False)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.split == 'hq':
            X = PIL.Image.open(self.filename[index])
        else:
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        if self.task:
            return X, torch.eye(2, dtype=int)[target[self.task_idx]]
        return X, target  # torch.eye(2, dtype=int)[target]

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
