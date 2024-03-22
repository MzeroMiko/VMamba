"""Custom video dataloader for ImageNet-P dataset, which comes in .mp4."""

import cv2
# from skvideo.io import VideoCapture
# import skvideo.io
import torch
import torch.utils.data as data
from torchvision.datasets.folder import DatasetFolder
from PIL import Image

import os
import os.path
import sys



class VideoFolder(DatasetFolder):

    def __init__(self, root, transform=None, target_transform=None, loader=None):
        super(VideoFolder, self).__init__(
            root, loader, ['.mp4'], transform=transform, target_transform=target_transform)

        self.vids = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        # cap = VideoCapture(path)
        cap = cv2.VideoCapture(path)

        frames = []

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret: break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(Image.fromarray(frame)).unsqueeze(0))

        cap.release()

        return torch.cat(frames, 0), target
