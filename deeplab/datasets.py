import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import pdb
from PIL import Image
from deeplab.custom_transforms import ScaleNRotate_siam
from chainercv.datasets import SBDInstanceSegmentationDataset

def readFlow(name):

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)


class DavisSiameseMAMLSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, all_frames=False, crop_size=(321, 321), mean=(128, 128, 128)):
        self.all_frames = all_frames
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.videos = [(i_id.strip().split()[0], int(i_id.strip().split()[1])) for i_id in open(list_path)]
        self.files = []
        # for split in ["train", "trainval", "val"]:

        for vid, nFrames in self.videos:
            for i in range(0, 1):
                support_x = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, i)
                support_y = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, i)
                query_x=[]
                query_y=[]
                #query_x = np.array([])

                if self.all_frames:
                    for j in range(i, nFrames):#[i+1, i+3, i+5, i+7, i+9, i+11, i+13, i+15, i+17, i+19, i+21, i+23, i+31, i+33, i+35, i+37]:
                        if j >= nFrames:
                            continue
                        img_file_2 = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, j)
                        lbl_file_2 = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, j)
                        query_x.append(img_file_2)
                        query_y.append(lbl_file_2)
                else:
                    for j in range(random.choice([0,1,2,3,4,5,6,7,8,9]), nFrames, random.choice([10,20])):#[i+1, i+3, i+5, i+7, i+9, i+11, i+13, i+15, i+17, i+19, i+21, i+23, i+31, i+33, i+35, i+37]:
                        if j >= nFrames:
                            continue
                        img_file_2 = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, j)
                        lbl_file_2 = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, j)
                        query_x.append(img_file_2)
                        query_y.append(lbl_file_2)

                self.files.append({
                    "support_x": support_x,
                    "support_y": support_y,
                    "query_x": query_x,
                    "query_y": query_y,
                    "name": vid})

        if not max_iters == None:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        return datafiles["support_x"], datafiles["support_y"], datafiles["query_x"], datafiles["query_y"], datafiles["name"]

