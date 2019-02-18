#Working version
import os
import random
import cv2
import pdb
import numpy as np
import torch

import matplotlib.pyplot as plt

class ScaleNRotate_siam(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25)):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales

    def __call__(self, sample):

        img_1, lbl_1, img_2, lbl_2, flo = sample

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot1 = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2
            rot2 = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc1 = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
            sc2 = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot1 = self.rots[random.randint(0, len(self.rots))]
            sc1 = self.scales[random.randint(0, len(self.scales))]
            rot2 = self.rots[random.randint(0, len(self.rots))]
            sc2 = self.scales[random.randint(0, len(self.scales))]

        h, w = img_1.shape[:2]
        center = (w / 2, h / 2)
        assert(center != 0)  # Strange behaviour warpAffine
        M1 = cv2.getRotationMatrix2D(center, rot1, sc1)
        M2 = cv2.getRotationMatrix2D(center, rot2, sc2)

        flagval = cv2.INTER_CUBIC
        img_11 = cv2.warpAffine(img_1, M1, (w, h), flags=flagval)
        img_22 = cv2.warpAffine(img_2, M2, (w, h), flags=flagval)
        floo = None if flo is None else cv2.warpAffine(flo, M2, (w, h), flags=flagval)

        flagval = cv2.INTER_NEAREST
        lbl_11 = cv2.warpAffine(lbl_1, M1, (w, h), flags=flagval)
        lbl_22 = cv2.warpAffine(lbl_2, M2, (w, h), flags=flagval)
        return img_11, lbl_11, img_22, lbl_22, floo


class ScaleNRotate_siam_maml(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25)):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales

    def __call__(self, sample):

        img_1,lbl_1 = sample

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot1 = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2
            rot2 = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc1 = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
            sc2 = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot1 = self.rots[random.randint(0, len(self.rots))]
            sc1 = self.scales[random.randint(0, len(self.scales))]
            rot2 = self.rots[random.randint(0, len(self.rots))]
            sc2 = self.scales[random.randint(0, len(self.scales))]

        h, w = img_1.shape[:2]
        center = (w / 2, h / 2)
        assert(center != 0)  # Strange behaviour warpAffine
        M1 = cv2.getRotationMatrix2D(center, rot1, sc1)
        M2 = cv2.getRotationMatrix2D(center, rot2, sc2)

        flagval = cv2.INTER_CUBIC
        img_11 = cv2.warpAffine(img_1, M1, (w, h), flags=flagval)

        flagval = cv2.INTER_NEAREST
        lbl_11 = cv2.warpAffine(lbl_1, M1, (w, h), flags=flagval)
        return img_11, lbl_11

class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        img_1 = sample['img1']
        img_2 = sample['img2']
        lbl_1 = sample['lbl1']
        lbl_2 = sample['lbl2']

        if random.random() < 0.5:
            img_11 = cv2.flip(img_1, flipCode=1)
            lbl_11 = cv2.flip(lbl_1, flipCode=1)

        if random.random() < 0.5:
            img_22 = cv2.flip(img_2, flipCode=1)
            lbl_22 = cv2.flip(lbl_2, flipCode=1)

        sample = {'img11':img_11, 'img22':img_22, 'lbl11':lbl_11, 'lbl22':lbl_22}

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W

            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(tmp)

        return sample
