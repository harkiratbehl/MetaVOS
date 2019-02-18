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


class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
	        self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name


class VOCDataSetTest(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, mean=(128, 128, 128), crop_size=(321, 321), ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
	        self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.crop_h, self.crop_w), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (self.crop_h, self.crop_w), interpolation=cv2.INTER_NEAREST)
        size = image.shape
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        image -= self.mean

        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name


class DavisSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.videos = [(i_id.strip().split()[0], int(i_id.strip().split()[1])) for i_id in open(list_path)]
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for vid, nFrames in self.videos:
            for i in range(nFrames):
                img_file_1 = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, i)
                lbl_file_1 = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, i)
                self.files.append({
                    "img_1": img_file_1,
                    "lab_1": lbl_file_1,
                    "name": (vid, i)})
        if not max_iters == None:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image_1 = cv2.imread(datafiles["img_1"], cv2.IMREAD_COLOR)
        label_1 = cv2.imread(datafiles["lab_1"], cv2.IMREAD_GRAYSCALE)

        image_1 = cv2.resize(image_1, (self.crop_h, self.crop_w), interpolation=cv2.INTER_CUBIC)
        label_1 = cv2.resize(label_1, (self.crop_h, self.crop_w), interpolation=cv2.INTER_NEAREST)

        name = datafiles["name"]
        if label_1.max()==255:
            label_1 = label_1 / 255
        size = image_1.shape

        #if self.scale:
        #    image_1, label_1 = self.generate_scale_label(image_1, label_1)

        image_1 = np.asarray(image_1, np.float32)
        image_1 -= self.mean

        '''
        img_h, img_w = label_1.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image_1 = np.asarray(image_1[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label_1 = np.asarray(label_1[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        '''

        #image = image[:, :, ::-1]  # change to BGR
        image_1 = image_1.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image_1 = image_1[:, :, ::flip]
            label_1 = label_1[:, ::flip]

        return image_1.copy(), label_1.copy(), np.array(size), name


class DavisSetTest(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.videos = [(i_id.strip().split()[0], int(i_id.strip().split()[1])) for i_id in open(list_path)]
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for vid, nFrames in self.videos:
            for i in range(nFrames):
                img_file_1 = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, i)
                lbl_file_1 = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, i)
                self.files.append({
                    "img_1": img_file_1,
                    "lab_1": lbl_file_1,
                    "name": (vid, i)})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image_1 = cv2.imread(datafiles["img_1"], cv2.IMREAD_COLOR)
        label_1 = cv2.imread(datafiles["lab_1"], cv2.IMREAD_GRAYSCALE)

        image_1 = cv2.resize(image_1, (self.crop_h, self.crop_w), interpolation=cv2.INTER_CUBIC)
        label_1 = cv2.resize(label_1, (self.crop_h, self.crop_w), interpolation=cv2.INTER_NEAREST)
        name = datafiles["name"]
        if label_1.max()==255:
            label_1 = label_1 / 255
        size = image_1.shape
        image_1 = np.asarray(image_1, np.float32)
        image_1 -= self.mean
        #image = image[:, :, ::-1]  # change to BGR
        image_1 = image_1.transpose((2, 0, 1))
        return image_1.copy(), label_1.copy(), np.array(size), name


class DavisSiameseSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.videos = [(i_id.strip().split()[0], int(i_id.strip().split()[1])) for i_id in open(list_path)]
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for vid, nFrames in self.videos:
            for i in range(nFrames):
                for j in range(i, nFrames):#[i+1, i+3, i+5, i+7, i+9, i+11, i+13, i+15, i+17, i+19, i+21, i+23, i+31, i+33, i+35, i+37]:
                    if j >= nFrames:
                        continue
                    img_file_1 = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, i)
                    lbl_file_1 = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, i)
                    img_file_2 = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, j)
                    lbl_file_2 = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, j)
                    flo_file_2 = self.root + "/Flows/480p/{}/{:05d}.flo".format(vid, j) #TDO
                    self.files.append({
                        "img_1": img_file_1,
                        "lab_1": lbl_file_1,
                        "img_2": img_file_2,
                        "lab_2": lbl_file_2,
                        "flo_2": flo_file_2,
                        "name": (vid, i, j)})
        if not max_iters == None:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image_1 = cv2.imread(datafiles["img_1"], cv2.IMREAD_COLOR)
        label_1 = Image.open(datafiles["lab_1"])
        image_2 = cv2.imread(datafiles["img_2"], cv2.IMREAD_COLOR)
        label_2 = Image.open(datafiles["lab_2"])
        flow_2 = readFlow(datafiles["flo_2"])

        label_1 = np.array(label_1, dtype=np.uint8)
        label_2 = np.array(label_2, dtype=np.uint8)
        size = image_1.shape

        if not image_1.shape[1]==854:
            image_1 = cv2.resize(image_1, (854, 480), interpolation=cv2.INTER_CUBIC)
            label_1 = cv2.resize(label_1, (854, 480), interpolation=cv2.INTER_NEAREST)
            image_2 = cv2.resize(image_2, (854, 480), interpolation=cv2.INTER_CUBIC)
            label_2 = cv2.resize(label_2, (854, 480), interpolation=cv2.INTER_NEAREST)
            flow_2 = cv2.resize(flow_2, (854, 480), interpolation=cv2.INTER_CUBIC)

        name = datafiles["name"]

        #if self.scale:
        #    image_1, label_1 = self.generate_scale_label(image_1, label_1)

        image_1 = np.asarray(image_1, np.float32)
        image_2 = np.asarray(image_2, np.float32)
        image_1 -= self.mean
        image_2 -= self.mean

        do_transform = np.random.choice(2) * 2 - 1
        if do_transform:
            scaling_rotation = ScaleNRotate_siam(rots=(-30, 30), scales=(0.75, 1.25))
            image_1, label_1, image_2, label_2, flow_2 = scaling_rotation((image_1, label_1, image_2, label_2, flow_2))

        #image = image[:, :, ::-1]  # change to BGR
        image_1 = image_1.transpose((2, 0, 1))
        image_2 = image_2.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image_1 = image_1[:, :, ::flip]
            label_1 = label_1[:, ::flip]
            image_2 = image_2[:, :, ::flip]
            label_2 = label_2[:, ::flip]
            flow_2 = flow_2[:, ::flip]
            
        return image_1.copy(), label_1.copy(), image_2.copy(), label_2.copy(), flow_2.copy(), np.array(size), name


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


class DavisSiameseSSL(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.images = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.images = self.images * int(np.ceil(float(max_iters) / len(self.images)))
        self.files = []

        for img in self.images:
            support_x = os.path.join(self.root, img)
            query_x = os.path.join(self.root, img)
            self.files.append({"support_x": support_x, "query_x": query_x})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        return datafiles["support_x"], datafiles["query_x"]


class DavisSiameseSetTestScribbles(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.videos = [(i_id.strip().split()[0], int(i_id.strip().split()[1])) for i_id in open(list_path)]
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for vid, nFrames in self.videos:
            for i in range(1):
                for j in range(i, nFrames):
                    if j >= nFrames:
                        continue
                    img_file_1 = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, i)
                    lbl_file_1 = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, i)
                    #lbl_file_1 = self.root + "/Squiggles/{}.png".format(vid)
                    img_file_2 = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, j)
                    lbl_file_2 = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, j)
                    flo_file_2 = self.root + "/Flows/480p/{}/{:05d}.flo".format(vid, j)
                    self.files.append({
                        "img_1": img_file_1,
                        "lab_1": lbl_file_1,
                        "img_2": img_file_2,
                        "lab_2": lbl_file_2,
                        "flo_2": flo_file_2,
                        "name": (vid, i, j)})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image_1 = cv2.imread(datafiles["img_1"], cv2.IMREAD_COLOR)
        label_1 = Image.open(datafiles["lab_1"])
        image_2 = cv2.imread(datafiles["img_2"], cv2.IMREAD_COLOR)
        label_2 = Image.open(datafiles["lab_2"])
        flow_2 = readFlow(datafiles["flo_2"])

        label_1 = np.array(label_1, dtype=np.uint8)
        label_2 = np.array(label_2, dtype=np.uint8)
        size = image_1.shape

        if not image_1.shape[1]==854:
            image_1 = cv2.resize(image_1, (854, 480), interpolation=cv2.INTER_CUBIC)
            label_1 = cv2.resize(label_1, (854, 480), interpolation=cv2.INTER_NEAREST)
            image_2 = cv2.resize(image_2, (854, 480), interpolation=cv2.INTER_CUBIC)
            label_2 = cv2.resize(label_2, (854, 480), interpolation=cv2.INTER_NEAREST)
            flow_2 = cv2.resize(flow_2, (854, 480), interpolation=cv2.INTER_CUBIC)

        name = datafiles["name"]

        #if self.scale:
        #    image_1, label_1 = self.generate_scale_label(image_1, label_1)

        image_1 = np.asarray(image_1, np.float32)
        image_2 = np.asarray(image_2, np.float32)
        image_1 -= self.mean
        image_2 -= self.mean

        #image = image[:, :, ::-1]  # change to BGR
        image_1 = image_1.transpose((2, 0, 1))
        image_2 = image_2.transpose((2, 0, 1))

        return image_1.copy(), label_1.copy(), image_2.copy(), label_2.copy(), flow_2.copy(), np.array(size), name


class DavisSiameseSetTestScribbles_val(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.videos = [(i_id.strip().split()[0], int(i_id.strip().split()[1])) for i_id in open(list_path)]
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for vid, nFrames in self.videos:
            if vid not in ['bmx-trees', 'car-shadow', 'drift-chicane', 'lab-coat']: 
                continue
            for i in range(1):
                for j in range(i, nFrames):
                    if j >= nFrames:
                        continue
                    img_file_1 = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, i)
                    lbl_file_1 = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, i)
                    #lbl_file_1 = self.root + "/Squiggles/{}.png".format(vid)
                    img_file_2 = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, j)
                    lbl_file_2 = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, j)
                    flo_file_2 = self.root + "/Flows/480p/{}/{:05d}.flo".format(vid, j)
                    self.files.append({
                        "img_1": img_file_1,
                        "lab_1": lbl_file_1,
                        "img_2": img_file_2,
                        "lab_2": lbl_file_2,
                        "flo_2": flo_file_2,
                        "name": (vid, i, j)})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image_1 = cv2.imread(datafiles["img_1"], cv2.IMREAD_COLOR)
        label_1 = Image.open(datafiles["lab_1"])
        image_2 = cv2.imread(datafiles["img_2"], cv2.IMREAD_COLOR)
        label_2 = Image.open(datafiles["lab_2"])
        flow_2 = readFlow(datafiles["flo_2"])

        label_1 = np.array(label_1, dtype=np.uint8)
        label_2 = np.array(label_2, dtype=np.uint8)
        size = image_1.shape

        if not image_1.shape[1]==854:
            image_1 = cv2.resize(image_1, (854, 480), interpolation=cv2.INTER_CUBIC)
            label_1 = cv2.resize(label_1, (854, 480), interpolation=cv2.INTER_NEAREST)
            image_2 = cv2.resize(image_2, (854, 480), interpolation=cv2.INTER_CUBIC)
            label_2 = cv2.resize(label_2, (854, 480), interpolation=cv2.INTER_NEAREST)
            flow_2 = cv2.resize(flow_2, (854, 480), interpolation=cv2.INTER_CUBIC)

        name = datafiles["name"]

        #if self.scale:
        #    image_1, label_1 = self.generate_scale_label(image_1, label_1)

        image_1 = np.asarray(image_1, np.float32)
        image_2 = np.asarray(image_2, np.float32)
        image_1 -= self.mean
        image_2 -= self.mean

        #image = image[:, :, ::-1]  # change to BGR
        image_1 = image_1.transpose((2, 0, 1))
        image_2 = image_2.transpose((2, 0, 1))

        return image_1.copy(), label_1.copy(), image_2.copy(), label_2.copy(), flow_2.copy(), np.array(size), name


class YoutubeObjSiameseSetTest(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, all_frames=False, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        file_images = open(list_path)
        self.files = []
        # for split in ["train", "trainval", "val"]:
        init_class_vid = None
        init_vid = None
        init_i = None
        for currentVideoLine in file_images:
            splitCurrentLine = currentVideoLine.strip().split('/')
            class_vid = splitCurrentLine[1]
            vid = splitCurrentLine[3]
            j = splitCurrentLine[-1].split('.')[0]

            if init_class_vid != class_vid or init_vid != vid:
                if init_i != None:
                     self.files.append({
                    "img_1": img_file_1,
                    "lab_1": lbl_file_1,
                    "query_x": query_x,
                    "query_y": query_y,
                    "name": (init_class_vid+init_vid)})

                init_class_vid = class_vid
                init_vid = vid
                init_i = j
                img_file_1 = self.root + "/{}/data/{}/shots/001/images/{}.png".format(class_vid, vid, init_i)
                lbl_file_1 = self.root + "/{}/data/{}/shots/001/labels/{}.jpg".format(class_vid, vid, init_i)
                query_x= [img_file_1]
                query_y= [lbl_file_1]
            else:
                img_file_2 = self.root + "/{}/data/{}/shots/001/images/{}.png".format(class_vid, vid, j)
                lbl_file_2 = self.root + "/{}/data/{}/shots/001/labels/{}.jpg".format(class_vid, vid, j)
                query_x.append(img_file_2)
                query_y.append(lbl_file_2)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        ############################
        # image_1 = cv2.imread(datafiles["img_1"], cv2.IMREAD_COLOR)
        # label_1 = Image.open(datafiles["lab_1"])
        # image_2 = cv2.imread(datafiles["img_2"], cv2.IMREAD_COLOR)
        # label_2 = Image.open(datafiles["lab_2"])
        # #flow_2 = cv2.imread(datafiles["flo_2"], cv2.IMREAD_GRAYSCALE)

        # label_1 = np.array(label_1, dtype=np.uint8)
        # label_2 = np.array(label_2, dtype=np.uint8)
        # size = image_1.shape

        # if not image_1.shape[1]==854:
        #     image_1 = cv2.resize(image_1, (854, 480), interpolation=cv2.INTER_CUBIC)
        #     #label_1 = cv2.resize(label_1, (854, 480), interpolation=cv2.INTER_NEAREST)
        #     image_2_prime = cv2.resize(image_2, (854, 480), interpolation=cv2.INTER_CUBIC)
        #     #label_2 = cv2.resize(label_2, (854, 480), interpolation=cv2.INTER_NEAREST)
        #     #flow_2 = cv2.resize(flow_2, (854, 480), interpolation=cv2.INTER_CUBIC)

        # name = datafiles["name"]

        # #if self.scale:
        # #    image_1, label_1 = self.generate_scale_label(image_1, label_1)

        # image_1 = np.asarray(image_1, np.float32)
        # image_2_prime = np.asarray(image_2_prime, np.float32)
        # image_2 = np.asarray(image_2, np.float32)
        # image_1 -= self.mean
        # image_2_prime -= self.mean
        # image_2 -= self.mean

        # #image = image[:, :, ::-1]  # change to BGR
        # image_1 = image_1.transpose((2, 0, 1))
        # image_2_prime = image_2_prime.transpose((2, 0, 1))
        # image_2 = image_2.transpose((2, 0, 1))

        # return image_1.copy(), label_1.copy(), image_2_prime.copy(), label_2.copy(), image_2.copy(), np.array(size), name
        #################################

        return datafiles["img_1"], datafiles["lab_1"], datafiles["query_x"], datafiles["query_y"], datafiles["name"]

class DavisSiameseSetTestDevScribbles(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.videos = [(i_id.strip().split()[0], int(i_id.strip().split()[1])) for i_id in open(list_path)]
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for vid, nFrames in self.videos:
            for i in range(1):
                for j in range(i, nFrames):
                    if j >= nFrames:
                        continue
                    img_file_1 = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, i)
                    lbl_file_1 = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, i)
                    #lbl_file_1 = self.root + "/Squiggles/{}.png".format(vid)
                    img_file_2 = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, j)
                    flo_file_2 = self.root + "/Flows/480p/{}/{:05d}.flo".format(vid, j)
                    self.files.append({
                        "img_1": img_file_1,
                        "lab_1": lbl_file_1,
                        "img_2": img_file_2,
                        "flo_2": flo_file_2,
                        "name": (vid, i, j)})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image_1 = cv2.imread(datafiles["img_1"], cv2.IMREAD_COLOR)
        label_1 = Image.open(datafiles["lab_1"])
        image_2 = cv2.imread(datafiles["img_2"], cv2.IMREAD_COLOR)
        flow_2 = readFlow(datafiles["flo_2"])

        label_1 = np.array(label_1, dtype=np.uint8)
        size = image_1.shape

        if not image_1.shape[1]==854:
            image_1 = cv2.resize(image_1, (854, 480), interpolation=cv2.INTER_CUBIC)
            label_1 = cv2.resize(label_1, (854, 480), interpolation=cv2.INTER_NEAREST)
            image_2 = cv2.resize(image_2, (854, 480), interpolation=cv2.INTER_CUBIC)
            flow_2 = cv2.resize(flow_2, (854, 480), interpolation=cv2.INTER_CUBIC)

        name = datafiles["name"]

        #if self.scale:
        #    image_1, label_1 = self.generate_scale_label(image_1, label_1)

        image_1 = np.asarray(image_1, np.float32)
        image_2 = np.asarray(image_2, np.float32)
        image_1 -= self.mean
        image_2 -= self.mean

        #image = image[:, :, ::-1]  # change to BGR
        image_1 = image_1.transpose((2, 0, 1))
        image_2 = image_2.transpose((2, 0, 1))

        return image_1.copy(), label_1.copy(), image_2.copy(), flow_2.copy(), np.array(size), name


class SegTrackSiameseSetTestScribbles(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.videos = [(i_id.strip().split()[0], int(i_id.strip().split()[1])) for i_id in open(list_path)]
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for vid, nFrames in self.videos:
            for i in range(1):
                for j in range(i, nFrames):
                    if j >= nFrames:
                        continue
                    img_file_1 = self.root + "/JPEGImages/{}/{:05d}.png".format(vid, i)
                    lbl_file_1 = self.root + "/GroundTruth/{}/{:05d}.png".format(vid, i)
                    #lbl_file_1 = self.root + "/Squiggles/{}.png".format(vid)
                    img_file_2 = self.root + "/JPEGImages/{}/{:05d}.png".format(vid, j)
                    lbl_file_2 = self.root + "/GroundTruth/{}/{:05d}.png".format(vid, j)
                    flo_file_2 = self.root + "/Flows/{}/{:05d}_flow.png".format(vid, j)
                    self.files.append({
                        "img_1": img_file_1,
                        "lab_1": lbl_file_1,
                        "img_2": img_file_2,
                        "lab_2": lbl_file_2,
                        "flo_2": flo_file_2,
                        "name": (vid, i, j)})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image_1 = cv2.imread(datafiles["img_1"], cv2.IMREAD_COLOR)
        label_1 = cv2.imread(datafiles["lab_1"], cv2.IMREAD_GRAYSCALE)
        image_2 = cv2.imread(datafiles["img_2"], cv2.IMREAD_COLOR)
        label_2 = cv2.imread(datafiles["lab_2"], cv2.IMREAD_GRAYSCALE)
        flow_2 = cv2.imread(datafiles["flo_2"], cv2.IMREAD_GRAYSCALE)

        label_1 = np.array(label_1, dtype=np.uint8)
        label_2 = np.array(label_2, dtype=np.uint8)
        size = image_1.shape

        if not image_1.shape[1]==854:
            image_1 = cv2.resize(image_1, (854, 480), interpolation=cv2.INTER_CUBIC)
            label_1 = cv2.resize(label_1, (854, 480), interpolation=cv2.INTER_NEAREST)
            image_2 = cv2.resize(image_2, (854, 480), interpolation=cv2.INTER_CUBIC)
            label_2 = cv2.resize(label_2, (854, 480), interpolation=cv2.INTER_NEAREST)
            flow_2 = cv2.resize(flow_2, (854, 480), interpolation=cv2.INTER_CUBIC)

        name = datafiles["name"]

        #if self.scale:
        #    image_1, label_1 = self.generate_scale_label(image_1, label_1)

        image_1 = np.asarray(image_1, np.float32)
        image_2 = np.asarray(image_2, np.float32)
        image_1 -= self.mean
        image_2 -= self.mean

        #image = image[:, :, ::-1]  # change to BGR
        image_1 = image_1.transpose((2, 0, 1))
        image_2 = image_2.transpose((2, 0, 1))

        return image_1.copy(), label_1.copy(), image_2.copy(), label_2.copy(), flow_2.copy(), np.array(size), name


class PascalSiameseSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True,
                 mirror=False, transform=False, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.do_transform = transform
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "lab": label_file,
                "name": name
            })
        if not max_iters == None:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image_1 = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label_1 = Image.open(datafiles["lab"])

        image_1 = np.asarray(image_1, np.float32)
        label_1 = np.array(label_1, dtype=np.uint8)

        image_1 = cv2.resize(image_1, (self.crop_h, self.crop_w), interpolation=cv2.INTER_CUBIC)
        label_1 = cv2.resize(label_1, (self.crop_h, self.crop_w), interpolation=cv2.INTER_NEAREST)

        image_2 = image_1.copy()
        label_2 = label_1.copy()
        size = image_1.shape
        name = datafiles["name"]

        image_1 = np.asarray(image_1, np.float32)
        image_2 = np.asarray(image_2, np.float32)
        image_1 -= self.mean
        image_2 -= self.mean

        do_transform = np.random.choice(2) * 2 - 1
        if self.do_transform and do_transform:
            scaling_rotation = ScaleNRotate_siam(rots=(-30, 30), scales=(0.75, 1.25))
            image_1, label_1, image_2, label_2, _ = scaling_rotation((image_1, label_1, image_2, label_2, None))

        #image = image[:, :, ::-1]  # change to BGR
        image_1 = image_1.transpose((2, 0, 1))
        image_2 = image_2.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image_1 = image_1[:, :, ::flip]
            label_1 = label_1[:, ::flip]
            image_2 = image_2[:, :, ::flip]
            label_2 = label_2[:, ::flip]

        return image_1.copy(), label_1.copy(), image_2.copy(), label_2.copy(), np.array(size), name



class PascalSiameseInstanceSet(data.Dataset):
    def __init__(self, root, split, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True,
                 mirror=False, transform=False, ignore_label=255):
        self.root = root
        self.list_path = osp.join(self.root, '{0}.txt'.format(split))
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.do_transform = transform
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.dataset = SBDInstanceSegmentationDataset(split=split)

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, 'img', name + '.jpg')
            
            #label_file, inst_img = self.dataset._load_label_inst(name)
            #print(label_file[np.nonzero(label_file > 0)])
            #cv2.imshow('a',255/20*inst_img)
            #cv2.waitKey()
            self.files.append({
                "img": img_file,
                "lab": name,
                "name": name
            })
        if not max_iters == None:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image_1 = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label_file, inst_file = self.dataset._load_label_inst(datafiles["lab"])
        label_1 = inst_file


        image_1 = np.asarray(image_1, np.float32)
        label_1 = np.array(label_1, dtype=np.uint8)

        image_1 = cv2.resize(image_1, (self.crop_h, self.crop_w), interpolation=cv2.INTER_CUBIC)
        label_1 = cv2.resize(label_1, (self.crop_h, self.crop_w), interpolation=cv2.INTER_NEAREST)

        image_2 = image_1.copy()
        label_2 = label_1.copy()
        size = image_1.shape
        name = datafiles["name"]

        image_1 = np.asarray(image_1, np.float32)
        image_2 = np.asarray(image_2, np.float32)
        image_1 -= self.mean
        image_2 -= self.mean

        do_transform = np.random.choice(2) * 2 - 1
        if self.do_transform and do_transform:
            scaling_rotation = ScaleNRotate_siam(rots=(-30, 30), scales=(0.75, 1.25))
            image_1, label_1, image_2, label_2, _ = scaling_rotation((image_1, label_1, image_2, label_2, None))

        #image = image[:, :, ::-1]  # change to BGR
        image_1 = image_1.transpose((2, 0, 1))
        image_2 = image_2.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image_1 = image_1[:, :, ::flip]
            label_1 = label_1[:, ::flip]
            image_2 = image_2[:, :, ::flip]
            label_2 = label_2[:, ::flip]

        return image_1.copy(), label_1.copy(), image_2.copy(), label_2.copy(), np.array(size), name


class VideoDataRNN(data.Dataset):
    def __init__(self, root, list_path, seq_len=2, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.mean = mean
        self.seq_len = seq_len
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.videos = [(i_id.strip().split()[0], int(i_id.strip().split()[1])) for i_id in open(list_path)]
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for vid, nFrames in self.videos:
            for i in range(nFrames-seq_len+1):
                batch = []
                img_file_0 = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, 0)
                lbl_file_0 = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, 0)
                batch.append({
                    "img": img_file_0,
                    "lab": lbl_file_0,
                    "name": (vid, i)})
                img_file = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, i)
                lbl_file = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, i)
                batch.append({
                    "img": img_file,
                    "lab": lbl_file})
                for j in range(i+1, i+seq_len):
                    img_file = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, j)
                    lbl_file = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, j)
                    flo_file = self.root + "/Flows/480p/{}/{:05d}.flo".format(vid, j)
                    batch.append({
                        "img": img_file,
                        "lab": lbl_file,
                        "flo": flo_file})
                self.files.append(tuple(batch))
        if not max_iters == None:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        flip = 1
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
        datafiles = self.files[index]
        name = datafiles[0]["name"]
        images = []
        labels = []
        flows  = []
        for i in range(self.seq_len+1):
            img = cv2.imread(datafiles[i]["img"], cv2.IMREAD_COLOR)
            size = img.shape
            img = np.asarray(img, np.float32) - self.mean
            img = img.transpose((2, 0, 1))
            img = img[:, :, ::flip]
            images.append(img.copy())
            lbl = np.array(Image.open(datafiles[i]["lab"]), dtype=np.uint8)
            lbl[lbl > 1] = 1
            lbl = lbl[:, ::flip]
            labels.append(lbl.copy())
            if i >= 2:
                flo = readFlow(datafiles[i]["flo"])
                flo = flo[:, ::flip, :] # Flow dimensions: [H, W, 2]
                flows.append(flo)
        images = np.stack(images, axis=0)
        labels = np.stack(labels, axis=0)
        flows  = np.stack(flows , axis=0)

        return images.copy(), labels.copy(), flows.copy(), np.array(size), name


class VideoTestDataRNN(data.Dataset):
    def __init__(self, root, list_path, mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.videos = [(i_id.strip().split()[0], int(i_id.strip().split()[1])) for i_id in open(list_path)]
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for vid, nFrames in self.videos:
            for j in range(nFrames):
                img_file = self.root + "/JPEGImages/480p/{}/{:05d}.jpg".format(vid, j)
                lbl_file = self.root + "/Annotations/480p/{}/{:05d}.png".format(vid, j)
                flo_file = self.root + "/Flows/480p/{}/{:05d}.flo".format(vid, j)
                self.files.append({
                        "img": img_file,
                        "lab": lbl_file,
                        "flo": flo_file,
                        "name": (vid, j)})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]

        img = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = img.shape
        img = np.asarray(img, np.float32) - self.mean
        img = img.transpose((2, 0, 1))
        lbl = np.array(Image.open(datafiles["lab"]), dtype=np.uint8)
        lbl[lbl > 1] = 1
        flo = np.zeros((size[0], size[1], 2))
        if name[1] > 0:
            flo = readFlow(datafiles["flo"])

        return img.copy(), lbl.copy(), flo.copy(), np.array(size), name

