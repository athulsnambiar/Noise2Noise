from torch.utils.data import Dataset
import cv2
import numpy as np
from typing import List
import os
import sys


# Constants
IMG_EXTS = [".jpg", ".png", ".jpeg"]



class NoiseDataset(Dataset):
    """Noise images dataset."""

    def __init__(self, path: str,
                 count = 30,
                 noisetype = "gaussian",
                 mean = 0.0,
                 std = 1.0,
                 seed = 0):
        """Initialize Class

        Arguments:
            path {str} -- Folder containing images

        Keyword Arguments:
            count {int} -- Number of Pairs of data (default: {30})
            noisetype {str} -- Type of noise (default: {"gaussian"})
            mean {float} -- Mean of noise if gaussian noise selected
                          (default: {0})
            std {float} -- Mean of noise if gaussian noise selected
                           (default: {1})
            seed {int} -- seed for numpy random generator (default: {0})
        """

        self.clean_imgs: List[np.ndarray] = []

        if os.path.isdir(path):
            for i in os.listdir(path):
                i_ = os.path.join(path, i)
                if (os.path.isfile(i_) and
                    os.path.splitext(i_)[1].lower() in IMG_EXTS):
                    # print(i)
                    img = cv2.imread(i_)
                    self.clean_imgs.append(img)
        else:
            print("Path doesn't exist...")
            sys.exit(0)
        self.clean_len = len(self.clean_imgs)
        self.count = count

        self.noisetype = noisetype
        if self.noisetype == "gaussian":
            self.mean = mean
            self.std = std

        np.random.seed(seed=seed)

    def __len__(self):
        return len(self.count)

    def __getitem__(self, idx):

        img = self.clean_imgs[int(idx)%self.clean_len]

        if self.noisetype == "gaussian":
            n1 = img + np.random.normal(self.mean, self.std,
                                        size=img.shape)
            n2 = img + np.random.normal(self.mean, self.std,
                                        size=img.shape)

        elif self.noisetype == "poisson":
            n1 = np.random.poisson(img)
            n2 = np.random.poisson(img)

        return n1, n2
