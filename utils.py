import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from scipy import misc


def imwrite_indexed(filename,array,color_palette):
  """ Save indexed png."""

  if np.atleast_3d(array).shape[2] != 1:
    raise Exception("Saving indexed PNGs requires 2D array.")

  im = Image.fromarray(array)
  im.putpalette(color_palette.ravel())
  
  # plt.figure()
  # plt.imshow(im)
  # plt.show()
  # im.show()
  # cv2.imshow('Result', np.array(im).astype(np.uint8))
  # cv2.waitKey(100)
  im.save(filename, format='PNG')


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

        img_1,lbl_1 = sample

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot1 = (self.rots[1] - self.rots[0]) * random.random() - (self.rots[1] - self.rots[0])/2
            sc1 = (self.scales[1] - self.scales[0]) * random.random() - (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot1 = self.rots[random.randint(0, len(self.rots))]
            sc1 = self.scales[random.randint(0, len(self.scales))]

        h, w = img_1.shape[:2]
        center = (w / 2, h / 2)
        assert(center != 0)  # Strange behaviour warpAffine
        M1 = cv2.getRotationMatrix2D(center, rot1, sc1)

        flagval = cv2.INTER_CUBIC
        img_11 = cv2.warpAffine(img_1, M1, (w, h), flags=flagval)

        flagval = cv2.INTER_NEAREST
        lbl_11 = cv2.warpAffine(lbl_1, M1, (w, h), flags=flagval)
        return img_11, lbl_11


def read_image_label(img, lab, mean, mirror=False, scale=False, rotate=False):

    image_1 = cv2.imread(img, cv2.IMREAD_COLOR)
    label_1 = Image.open(lab)

    label_1 = np.array(label_1, dtype=np.uint8)
    size = image_1.shape

    if not image_1.shape[1 ] ==854:
        image_1 = cv2.resize(image_1, (854, 480), interpolation=cv2.INTER_CUBIC)
        label_1 = cv2.resize(label_1, (854, 480), interpolation=cv2.INTER_NEAREST)

    # if self.scale:
    #    image_1, label_1 = self.generate_scale_label(image_1, label_1)

    image_1 = np.asarray(image_1, np.float32)
    image_1 -= mean

    do_transform = np.random.choice(2) * 2 - 1
    if scale and do_transform:
        theta = 30 if rotate else 0
        scaling_rotation = ScaleNRotate_siam(rots=(-theta, theta), scales=(0.75, 1.25))
        image_1, label_1 = scaling_rotation((image_1, label_1))

    # image = image[:, :, ::-1]  # change to BGR
    image_1 = image_1.transpose((2, 0, 1))
    if mirror:
        flip = np.random.choice(2) * 2 - 1
        image_1 = image_1[:, :, ::flip]
        label_1 = label_1[:, ::flip]

    # THIS FUNCTION IS WORKING FINE
    image_1 = image_1[np.newaxis, ...]
    label_1 = label_1[np.newaxis, ...]

    return image_1.copy(), label_1.copy(), size

def read_image_label_segtrack(img, lab, mean, mirror=False, scale=False, rotate=False):
    image_1 = cv2.imread(img, cv2.IMREAD_COLOR)
    label_1_demo = cv2.imread(lab, cv2.IMREAD_COLOR)

    # image_1 = Image.open(img)
    # label_1_demo = Image.open(lab)

    label_1_demo = np.array(label_1_demo, dtype=np.uint8)

    label_2 = label_1_demo[:,:,0]

    label_1 = np.zeros((label_2.shape[0],label_2.shape[1]), dtype=int)
    label_1[label_2>=128] = 1
    # al = np.where(label_1==1)
    # print(np.where(label_1==1))
    # print(zip(al[0],al[1]))

    image_1 = np.asarray(image_1, np.float32)
    size = image_1.shape
    

    if not image_1.shape[1 ] ==854:
        image_1 = cv2.resize(image_1, (854, 480), interpolation=cv2.INTER_CUBIC)
        label_1 = cv2.resize(label_1, (854, 480), interpolation=cv2.INTER_NEAREST)

    # if self.scale:
    #    image_1, label_1 = self.generate_scale_label(image_1, label_1)

    
    image_1 -= mean

    do_transform = np.random.choice(2) * 2 - 1
    if scale and do_transform:
        theta = 30 if rotate else 0
        scaling_rotation = ScaleNRotate_siam(rots=(-theta, theta), scales=(0.75, 1.25))
        image_1, label_1 = scaling_rotation((image_1, label_1))

    # image = image[:, :, ::-1]  # change to BGR
    image_1 = image_1.transpose((2, 0, 1))
    if mirror:
        flip = np.random.choice(2) * 2 - 1
        image_1 = image_1[:, :, ::flip]
        label_1 = label_1[:, ::flip]

    # THIS FUNCTION IS WORKING FINE
    image_1 = image_1[np.newaxis, ...]
    label_1 = label_1[np.newaxis, ...]

    return image_1.copy(), label_1.copy(), size

def read_image_only(img, lab, mean, RESIZE=(321,321), mirror=False, scale=False, rotate=False):

    image_1 = cv2.imread(img, cv2.IMREAD_COLOR)

    size = image_1.shape

    # if not image_1.shape[1 ] ==854:
    image_1 = cv2.resize(image_1, (RESIZE[1], RESIZE[0]), interpolation=cv2.INTER_CUBIC)

    # if self.scale:
    #    image_1, label_1 = self.generate_scale_label(image_1, label_1)

    label_1= np.zeros(size)
    image_1 = np.asarray(image_1, np.float32)
    image_1 -= mean

    do_transform = np.random.choice(2) * 2 - 1
    if scale and do_transform:
        theta = 30 if rotate else 0
        scaling_rotation = ScaleNRotate_siam(rots=(-theta, theta), scales=(0.75, 1.25))
        image_1, label_1 = scaling_rotation((image_1, label_1))

    # image = image[:, :, ::-1]  # change to BGR
    image_1 = image_1.transpose((2, 0, 1))
    if mirror:
        flip = np.random.choice(2) * 2 - 1
        image_1 = image_1[:, :, ::flip]

    # THIS FUNCTION IS WORKING FINE
    image_1 = image_1[np.newaxis, ...]

    return image_1.copy(), size


def gen_bbox_new(label, instance_list, enlarge=False, ratio=1.0):
    bbox = np.zeros((len(instance_list), 4), float)
    bbox_enlarge = 0.15 if enlarge else 0.0

    for i in instance_list:
        [x, y] = np.where(label[:, :] == i + 1)
        if len(y) > 0:
            y = sorted(y)
            x = sorted(x)
            wmin = y[int((len(y) - 1) * (1 - ratio))]
            wmax = y[int((len(y) - 1) * (ratio))] + 1
            hmin = x[int((len(x) - 1) * (1 - ratio))]
            hmax = x[int((len(x) - 1) * (ratio))] + 1
        else:
            bbox[i, :] = [0, 0, 1, 1]
            continue

        bbox_h = hmax - hmin
        bbox_w = wmax - wmin

        wmin = np.clip((wmin - bbox_enlarge * bbox_w), 0, label.shape[1] - 1)
        wmax = np.clip((wmax + bbox_enlarge * bbox_w), wmin + 1, label.shape[1])
        hmin = np.clip((hmin - bbox_enlarge * bbox_h), 0, label.shape[0] - 1)
        hmax = np.clip((hmax + bbox_enlarge * bbox_h), hmin + 1, label.shape[0])

        bbox[i, :] = [int(wmin), int(hmin), int(wmax), int(hmax)]

    return bbox.astype(int)