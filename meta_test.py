import argparse

import pickle
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc as sm
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
from deeplab.model import ResNetUpsampled_Deeplab
from deeplab.loss import CrossEntropy2d
from deeplab.datasets import DavisSiameseMAMLSet
import matplotlib.pyplot as plt
import random
from PIL import Image
import pdb
import timeit

from datetime import datetime
from tqdm import tqdm
from deeplab.clustering import clustering, clustering_morph
import copy
import gc
from sklearn.cluster import MiniBatchKMeans

from src.disp import labelcolormap
from src.bbox import gen_bbox, label_to_prob, combine_prob, prob_to_label, IoU
from src.get_params import *
from src.loss_func import binary_cross_entropy2d, cross_entropy2d
from src.label_transfer import label_transfer, apply_crf
from src.utils import read_image_label, imwrite_indexed, read_image_only, gen_bbox_new

colors = labelcolormap(500)

start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

BATCH_SIZE = 1
DATA_DIRECTORY = 'DAVIS'
VERSION = '2017'
SPLIT = 'val'
DATA_LIST_PATH = 'dataset/{}_videos_{}.txt'.format(SPLIT, VERSION)
IGNORE_LABEL = 255
INPUT_SIZE = '240,427'
LEARNING_RATE = 2.5e-6
MOMENTUM = 0.9
NUM_CLASSES = 2
NUM_FT_STEPS = 150001
POWER = 0.9
# RESTORE_FROM = 'random'
# RESTORE_FROM = './dataset/MS_DeepLab_resnet_pretrained_COCO_init.pth'
RESTORE_FROM = './snapshots/DAVIS_{}_prototypical_MODES_train_max_109000.pth'.format(VERSION)
# RESTORE_FROM = './snapshots/DAVIS_{}_train_on_pascal_1000.pth'.format(VERSION)
#RESTORE_FROM = './snapshots/Pascal_train_similarityloss_bg_50000.pth'
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

FOR_BAC = 0
WARP = 0
COMBINE = 0
TRACKING = 0
# all three of above have to be made true.

COMBINE = 0  # for naive tracking
DISPLAY = 0
MEASURE = 1

TRACK_BASIC = 0
###########################################################################################
PROTOTYPICAL_SINGLE_MODE = False  # If False, then it uses NUMBER_OF_CLUSTERS modes
number_of_clusters = 200
NUMBER_OF_CLUSTERS_train = 10
BG_Factor = 4

CC_Outlier_Removal = True
Num_FIX_Iterations = 2

ONLINE_UPDATE = True
Update_Frequency = 5  # Every "Update_Frequency" frame we do an online update.
Mode_Update_Simularity_Threshold = 0.5
MODES_ADDED = 10
global NUMBER_OF_CLUSTERS
###########################################################################################


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_FT_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=1,
                        help="choose gpu device.")
    parser.add_argument("--cluster-num", type=int, default=number_of_clusters,
                        help="num of clusters.")
    return parser.parse_args()


args = get_arguments()


def flip(x, dim):
    if x.is_cuda:
        return torch.index_select(x, dim,
                                  torch.arange(x.size(dim) - 1, -1, -1).long().to(torch.device('cuda:{}'.format(1))))
    else:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long())


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 2


def adjust_learning_rate_step(optimizer):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    optimizer.param_groups[0]['lr'] = 0.2 * optimizer.param_groups[0]['lr']
    optimizer.param_groups[1]['lr'] = 0.2 * optimizer.param_groups[1]['lr']


def getBB(mask, margin):
    if mask.max() == 0:
        return np.zeros_like(mask)
    coords = cv2.boundingRect(mask.astype(np.uint8))
    H, W = mask.shape
    xmin = (coords[1] - margin) if (coords[1] - margin) > 0 else 0
    ymin = (coords[0] - margin) if (coords[0] - margin) > 0 else 0
    xmax = (coords[1] + coords[3] + margin) if (coords[1] + coords[3] + margin) < H else (H - 1)
    ymax = (coords[0] + coords[2] + margin) if (coords[0] + coords[2] + margin) < W else (W - 1)
    bb = np.zeros_like(mask)
    bb[xmin:xmax, ymin:ymax] = 1
    return bb


def main():

    NUMBER_OF_CLUSTERS = args.cluster_num
    # copy_as = colors[1:]
    # np.random.shuffle(copy_as)
    # colors[1:] = copy_as
    """Create the model and start the training."""
    SEED = 136
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    H, W = map(int, args.input_size.split(','))
    input_size = (H, W)

    cudnn.enabled = True
    gpu = args.gpu
    device = torch.device('cuda:{}'.format(gpu))

    # Create network.
    emb_size = (128, H, W)
    model_base = ResNetUpsampled_Deeplab(emb_size)
    # For a small batch size, it is better to keep
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model.
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    if RESTORE_FROM == 'random':
        print('\nProceeding with random initialization...\n')
    elif RESTORE_FROM == './dataset/MS_DeepLab_resnet_pretrained_COCO_init.pth':
        saved_state_dict = torch.load(args.restore_from)
        new_params = model_base.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if i_parts[0] == 'Scale':
                i_parts = i_parts[1:]
            if args.num_classes == 21 or not i_parts[0] == 'layer5':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
        model_base.load_state_dict(new_params)
    else:
        saved_state_dict = torch.load(args.restore_from, map_location={'cuda:3' : 'cuda:{}'.format(gpu), 'cuda:2' : 'cuda:{}'.format(gpu), 'cuda:1' : 'cuda:{}'.format(gpu), 'cuda:0' : 'cuda:{}'.format(gpu)})
        model_base.load_state_dict(saved_state_dict)

    # model.float()
    # model.eval() # use_global_stats = True
    model_base.eval()

    cudnn.benchmark = True


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    testloader = data.DataLoader(
        DavisSiameseMAMLSet(args.data_dir, args.data_list, all_frames=True, crop_size=input_size, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    fin = open('logs/test_{}_'.format(VERSION) + str(datetime.now().time()), 'w+')

    for i_iter, batch in enumerate(testloader):
        support_x, support_y, query_x, query_y, name = batch
        # if name[0] not in ['dogs-jump','dog','cows','camel']:#''dog', 'dogs-jump', 'drift-chicane', 'drift-straight', 'soapbox']:
        #     continue
        nFrames = len(query_x)

        #########################################################
        ### Reading the first frame:
        ###################
        img_s = support_x[0]
        lab_s = support_y[0]
        supp_image, supp_label, size = read_image_label(img_s, lab_s, mean=IMG_MEAN, mirror=False, scale=False, rotate=False)
        if VERSION == '2016':
            supp_label[supp_label > 0] = 1  # Single object segmentation

        last_seen = np.zeros((supp_label.max(), supp_label.shape[1], supp_label.shape[2]))
        for l in range(supp_label.max()):
            last_seen[l] = (supp_label == (l+1)).astype(np.float)

        supp_label = cv2.resize(supp_label[0], (emb_size[2], emb_size[1]), interpolation=cv2.INTER_NEAREST)[np.newaxis, :, :]
        supp_image = torch.from_numpy(supp_image)
        supp_label = torch.from_numpy(supp_label)
        # supp_label[supp_label > 0] = 1 #Single object segmentation; we ignore other labels and reduce everything to one.
        supp_label = supp_label.long()
        cls1 = np.unique(supp_label[supp_label >= 0].numpy())
        instance_num = cls1.max()
        #########################################################


        model = copy.deepcopy(model_base)
        model.to(device)

        #########################################################
        ### Compute a total loss for the rest of the video frames:
        ###################
        nCls = cls1.shape[0]
        mean_ious_seq = np.zeros(nCls)
        mean_ious_seq_len = np.zeros(nCls)
        video_loss = 0
        video_loss_neg = 0






        if 0:
            print('----------------------------------------------------------------')

            optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.learning_rate},
                                   {'params': get_10x_lr_params(model), 'lr': args.learning_rate}],
                                  lr=args.learning_rate, momentum=args.momentum, weight_decay=0.0005)#args.weight_decay)
            optimizer.zero_grad()

            NUM_FT_STEPS = 10
            for ft_iter in range(NUM_FT_STEPS):

                img_q = query_x[0][0]
                lab_q = query_y[0][0]
                query_image, query_label, _ = read_image_label(img_q, lab_q, mean=IMG_MEAN, mirror=False, scale=False, rotate=False)
                if VERSION=='2016':
                    query_label[query_label > 0] = 1  # Single object segmentation
                gt = query_label[0].copy().astype(np.int16)
                query_label = cv2.resize(query_label[0], (emb_size[2], emb_size[1]), interpolation=cv2.INTER_NEAREST)[np.newaxis, :, :]
                inmax = np.abs(query_image[0]).max(0)
                gt[inmax == 0] = -1
                inmax = cv2.resize(inmax, (emb_size[2], emb_size[1]), interpolation=cv2.INTER_NEAREST)
                query_label = query_label.astype(np.int16)
                query_label[0][inmax == 0] = -1 #To remove the void regions made by rotation augmentation
                query_label = torch.from_numpy(query_label).long()
                query_image = torch.from_numpy(query_image)
                cls2 = np.unique(query_label[query_label >= 0].numpy())
                if len(list(set(cls2)-set(cls1))) > 0: continue  # Skip the query frame if it has more labels than the support frame
                query_label_resh = query_label.view(-1).to(device)

                embeddings = model(query_image.to(device))
                embeddings = embeddings.view(1, emb_size[0], -1)

                if PROTOTYPICAL_SINGLE_MODE:
                    ####################################################################################
                    ### Prototypical networks, using one single prototype for each instance (inc. bg):
                    supp_label_resh = supp_label.view(1, -1)
                    prototypes = torch.stack([torch.mean(embedding_reference_resh[:, (supp_label_resh == float(l))[0]], dim=-1) for l in cls1])
                    prototypes = prototypes.unsqueeze(0).to(device)
                    DO_NORMALIZATION = 0
                    if DO_NORMALIZATION:
                        prototypes = prototypes / torch.norm(prototypes, dim=-1).view(prototypes.size()[0], prototypes.size()[1], 1)
                        embeddings = embeddings / torch.norm(embeddings, dim=1).view(embeddings.size()[0], 1, embeddings.size()[-1])
                    y = torch.matmul(prototypes, embeddings)
                    y = F.softmax(y, dim=1)
                    y = y[0].permute(1, 0)
                    loss = F.cross_entropy(y, query_label_resh, ignore_index=-1, size_average=True) / float(NUM_FT_STEPS)
                    output_probs = y.cpu().data.numpy().copy()
                    ####################################################################################
                else:
                    ####################################################################################
                    ### Prototypical networks, using multiple modes/clusters to represent parts of each instance (inc. bg)
                    ### Using max-pool approach
                    cluster_map = mapr.copy()

                    modes = []
                    for l in cls1:
                        cc = 0
                        for c in range(N_CLUSTERS):
                            indices = (cluster_map == (N_CLUSTERS * l + c))
                            idx = np.where(indices.reshape(-1) > 0)[0]
                            if indices.size > 0:
                                cc += 1
                                cluster_embeddings = embedding_reference_resh[:, idx]
                                cluster_mean_embedding = cluster_embeddings.mean(dim=1)
                                modes.append(cluster_mean_embedding)
                        ### If number of computed clusters within an object is less than N_CLUSTERS,
                        ### then fill/replicate the rest of N_CLUSTER embeddings with previous clusters of that object:
                        for j in range(N_CLUSTERS - cc):
                            modes.append(modes[l * N_CLUSTERS + j])
                    modes = torch.stack(modes, dim=0).unsqueeze(0).to(device)

                    DO_NORMALIZATION = 1
                    if DO_NORMALIZATION:
                        modes = modes / torch.norm(modes, dim=-1).view(modes.size()[0], modes.size()[1], 1)
                        embeddings = embeddings / torch.norm(embeddings, dim=1).view(embeddings.size()[0], 1, embeddings.size()[-1])

                    ### Computing the similarity
                    y = torch.matmul(modes, embeddings)
                    y = F.softmax(y, dim=1)
                    y = y[0].permute(1, 0)
                    ### Extracting the maximum probability cluster of each class (To be used for evaluation, not in the loss):
                    output_probs = torch.stack([y[:, l*N_CLUSTERS:(l+1)*N_CLUSTERS].max(1)[0] for l in cls1], dim=-1).cpu().data.numpy().copy()
                    y_parts = []
                    for l in cls1:
                        y_part = y[:, l*N_CLUSTERS:(l+1)*N_CLUSTERS]
                        ### Taking argmax of the probabilities of the clusters of the correct class for each sample and
                        ### selecting the one with maximum score and computing the loss only for that cluster in each class:
                        ymaxidx = y_part.max(1)[1].unsqueeze(1).long()
                        ymaxidx_oh = 0 * torch.LongTensor(ymaxidx.size()[0], N_CLUSTERS).to(device)
                        ymaxidx_oh.scatter_(1, ymaxidx, 1)
                        y_part = y_part[ymaxidx_oh.byte()]
                        y_parts.append(y_part)
                    y = torch.stack(y_parts, dim=1)
                    y = y / (y.sum(1).unsqueeze(1))
                    loss = F.cross_entropy(y, query_label_resh, ignore_index=-1, size_average=True) / 1#float(NUM_FT_STEPS)

                    # '''
                    ####################################
                    ### Computing the negative loss:
                    ql_oh = 0 * torch.LongTensor(query_label_resh.size()[0], cls1.shape[0]).to(device)
                    ql_oh.scatter_(1, query_label_resh.unsqueeze(1), 1)
                    ### Negative loss is computed only for incorrect classes: (1-ql_oh)
                    loss_neg = -(1-y).clamp(min=1e-12).log() * (1-ql_oh).float()
                    loss_neg = loss_neg.sum(1).mean() / 1#float(NUM_FT_STEPS)

                    loss = loss + loss_neg
                    video_loss_neg += loss_neg.data
                    ####################################################################################
                    # '''
                loss.backward()
                video_loss += loss.data

                ####################
                ### Computing the accuracy:
                output = np.argmax(output_probs, axis=1).reshape(emb_size[1:])
                output = cv2.resize(output, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                M = np.zeros((nCls, nCls))
                for i in range(M.shape[0]):
                    outg = output[gt==i]
                    for j in range(M.shape[1]):
                        M[i, j] = (outg==j).astype(np.float).sum()

                for i in range(M.shape[0]):
                    den = (M[i, :].sum() + M[:, i].sum() - M[i, i])
                    if (gt==i).astype(np.float).sum()!=0:
                        mean_ious_seq[i] += M[i, i] / den
                        mean_ious_seq_len[i] += 1
                #########################################################

                # adjust_learning_rate(optimizer, i_iter)
                optimizer.step()

                print('{}: 1st frame: FT_iter_{}, mean-loss = {:.04f}, mean-neg-loss = {:.04f}, mean-iou = {:.04f}'.format(name[0], ft_iter, video_loss, video_loss_neg, np.mean(mean_ious_seq[1:] / mean_ious_seq_len[1:])))
                mean_ious_seq *= 0
                mean_ious_seq_len *= 0
                video_loss = 0
                video_loss_neg = 0

                #########################################################
                ### Computing the instance prototypes in the first frame:
                ###################
                embedding_reference = model(supp_image.to(device))
                supp_label_resh = supp_label.view(1, -1)
                embedding_reference_resh = embedding_reference.view(emb_size[0], -1)
                embedding_reference_resh /= torch.norm(embedding_reference_resh, dim=0).view(1, embedding_reference_resh.size()[-1])
                selected_pixels_list = [np.where(supp_label_resh.numpy() == l)[1] for l in cls1]
                if PROTOTYPICAL_SINGLE_MODE:
                    N_CLUSTERS = min([1] + [np.sum(supp_label.numpy()[0] == l) for l in cls1])
                else:
                    ## If number of pixels in the object is less than NUMBER_OF_CLUSTERS, N_CLUSTERS should be the lower value.
                    N_CLUSTERS = min([NUMBER_OF_CLUSTERS_train] + [np.sum(supp_label.numpy()[0] == l) for l in cls1])
                modes, mapr, class_labels_indcs = clustering(embedding_reference_resh.cpu().data.numpy(),
                                                             (supp_image[0].numpy().transpose(1, 2, 0) + IMG_MEAN).astype('uint8'),
                                                             selected_pixels_list, emb_size, nclusters=N_CLUSTERS,
                                                             method='kmeans', pca_comps=None)
                modes_numpy = modes.copy()
                #########################################################
                # pdb.set_trace()
                print('----------------------------------------------------------------')

            #########################################
            ### Garbage Collection
            del embeddings, embedding_reference, embedding_reference_resh, y, y_part, y_parts
            gc.collect()
            torch.cuda.empty_cache()
            #########################################
            mean_ious_seq *= 0
            mean_ious_seq_len *= 0



        with torch.no_grad():

            embedding_reference = model(supp_image.to(device))
            supp_label_resh = supp_label.view(1, -1)
            embedding_reference_resh = embedding_reference.view(emb_size[0], -1)
            embedding_reference_resh /= torch.norm(embedding_reference_resh, dim=0).view(1, embedding_reference_resh.size()[-1])
            selected_pixels_list = [np.where(supp_label_resh.numpy() == l)[1] for l in cls1]
            if PROTOTYPICAL_SINGLE_MODE:
                N_CLUSTERS = min([1] + [np.sum(supp_label.numpy()[0] == l) for l in cls1])
            else:
                ## If number of pixels in the object is less than NUMBER_OF_CLUSTERS, N_CLUSTERS should be the lower value.
                N_CLUSTERS = min([NUMBER_OF_CLUSTERS] + [np.sum(supp_label.numpy()[0] == l) for l in cls1])
            modes, mapr, class_labels_indcs = clustering(embedding_reference_resh.cpu().data.numpy(),
                                                         (supp_image[0].numpy().transpose(1, 2, 0) + IMG_MEAN).astype('uint8'),
                                                         selected_pixels_list, emb_size, nclusters=N_CLUSTERS,
                                                         bg_factor=BG_Factor, method='kmeans', pca_comps=None)
            modes_mean = modes.mean(1)

            number_of_modes = np.zeros(nCls).astype(np.int16)
            number_of_modes[0] = N_CLUSTERS * BG_Factor
            for i in range(1, nCls):
                number_of_modes[i] = N_CLUSTERS

            # pdb.set_trace()
            nanidx = np.where(np.isnan(modes_mean))[0]
            for x in nanidx:
                clsnan = (x - number_of_modes[0]) // N_CLUSTERS
                cc = 0
                if clsnan >= 0:
                    base = clsnan * N_CLUSTERS + number_of_modes[0]
                else:
                    base = 0
                while np.isnan(modes_mean[base + cc]):
                    cc += 1
                modes[x, :] = modes[base + cc, :]

            cumsum_modes = np.cumsum(number_of_modes)
            cumsum_modes = np.hstack([0, cumsum_modes])
            base_modes = []
            for i in range(nCls):
                class_modes = modes[cumsum_modes[i]:cumsum_modes[i+1], :]
                base_modes.append(class_modes)
            modes_base_list = copy.deepcopy(base_modes)


            for idx in tqdm(range(0, nFrames), total=nFrames, desc='Evaluating the video:{}'.format(name[0]), ncols=120,
                            leave=False):
                img_q = query_x[idx][0]
                lab_q = query_y[idx][0]
                if 1:
                    query_image, query_label, size = read_image_label(img_q, lab_q, mean=IMG_MEAN, mirror=False,
                                                                      scale=False, rotate=False)
                    # pdb.set_trace()
                    if VERSION == '2016':
                        query_label[query_label > 0] = 1  # Single object segmentation
                    query_label = torch.from_numpy(query_label)
                else:
                    query_image, size = read_image_only(img_q, lab_q, RESIZE=input_size, mean=IMG_MEAN, mirror=False, scale=False,
                                                        rotate=False)

                query_image = torch.from_numpy(query_image)
                embeddings = model(query_image.to(device))

                ##########################################
                ### Prototypical networks, using multiple modes/clusters to represent parts of each instance (inc. bg)
                ### Using max-pool approach
                modes_list = copy.deepcopy(modes_base_list)
                modes_numpy = np.vstack(modes_list)
                modes = torch.from_numpy(modes_numpy).unsqueeze(0).to(device)
                modes = modes / torch.norm(modes, dim=-1).view(modes.size()[0], modes.size()[1], 1)
                embeddings = embeddings.view(1, emb_size[0], -1)
                embeddings = embeddings / torch.norm(embeddings, dim=1).view(embeddings.size()[0], 1, embeddings.size()[-1])
                y = torch.matmul(modes, embeddings.view(1, emb_size[0], -1))
                y = F.softmax(y, dim=1)
                y = y[0].permute(1, 0)

                if 1:
                    output_probs = torch.stack([y[:, cumsum_modes[l]:cumsum_modes[l+1]].max(1)[0] for l in cls1], dim=-1)
                else:
                    topk = torch.topk(y, 20, dim=1)
                    y = torch.zeros_like(y).scatter_(1, topk[1], y)
                    output_probs = torch.stack([y[:, cumsum_modes[l]:cumsum_modes[l+1]].sum(1) for l in cls1], dim=-1)

                output_cluster_labels = torch.stack([y[:, :].max(1)[1]], dim=-1).cpu().data.numpy().copy()
                ##########################################

                #########################################
                ##### COMBINING PROP AND SIAM
                # output_indices = output_probs.max(1)
                output_cluster_labels_n = output_cluster_labels.reshape(emb_size[1:])


                ######################
                ### Apply CRF
                # output_probs = F.softmax(output_probs, dim=1)
                output_probs = output_probs / (output_probs.sum(1).unsqueeze(1))
                output_probs = output_probs.transpose(1, 0).view(1, cls1.shape[0], emb_size[1], emb_size[2])
                output_probs = F.interpolate(output_probs, (query_image.shape[2], query_image.shape[3]), mode='bilinear', align_corners=True)
                output_probs = output_probs.cpu().data.numpy()


                #############################################################################################
                #############################################################################################
                #############################################################################################
                #############################################################################################
                #############################################################################################
                ### Naive Outlier Removal Using Connected Components:
                if CC_Outlier_Removal:
                    for cc_iter in range(Num_FIX_Iterations):
                        output = np.argmax(output_probs[0], axis=0)
                        for l in range(nCls-1):
                            mask = (output == (l+1)).astype(np.uint8)
                            ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            comps_num, comps_labels, comps_stats, comp_cents = cv2.connectedComponentsWithStats(thresh, connectivity=4, ltype=cv2.CV_32S)
                            filt = np.ones_like(mask).astype(np.float)
                            for cc in range(1, comps_num):
                                if comps_stats[cc, -1] < 10:
                                    filt[comps_labels == cc] = 0.1
                                    continue
                                overlap = (comps_labels == cc).astype(np.float) * last_seen[l].astype(np.float)
                                if overlap.sum() == 0:
                                    filt[comps_labels == cc] = 0.1
                            # pdb.set_trace()
                            output_probs[0][l+1] *= filt
                #############################################################################################
                #############################################################################################
                #############################################################################################
                #############################################################################################
                #############################################################################################


                if os.path.isdir('output_probs_{}'.format(VERSION)) != True:
                    os.mkdir('output_probs_{}'.format(VERSION))
                if os.path.isdir('output_images_{}'.format(VERSION)) != True:
                    os.mkdir('output_images_{}'.format(VERSION))
                if os.path.isdir('output_probs_{}/{}'.format(VERSION, name[0])) != True:
                    os.mkdir('output_probs_{}/{}'.format(VERSION, name[0]))
                np.save('output_probs_{}/{}/{:05}.npy'.format(VERSION, name[0], idx), output_probs.astype(np.float16))


                # output = np.argmax(output_probs, axis=1).reshape(emb_size[1:])
                if 0:#idx != 0 and idx != (nFrames-1):
                    flow_dir = os.path.join(DATA_DIRECTORY, 'Flows', '480p', name[0])
                    flow = flo.readFlow(os.path.join(flow_dir, '%05d.flo' % (idx)))
                    output = apply_crf(output_probs, query_image, IMG_MEAN, flow_2=flow, crf_niter=1)
                else:
                    output = apply_crf(output_probs, query_image, IMG_MEAN, flow_2=None, crf_niter=0)
                ######################

                # output_cluster_labels =
                output1 = output
                if idx != 0:
                    if WARP or FOR_BAC:
                        output1 = combine(output1, warp_label, bbox, idx)
                        if DISPLAY:
                            show_frame(output1, idx, bbox)
                    else:
                        if COMBINE == 2:
                            bbox = gen_bbox(prev_output, range(instance_num), True)
                            output1 = combine(output1, prev_output, bbox, idx)
                            if DISPLAY:
                                image = cv2.imread(img_q)
                                show_frame_new(image, output1, idx, bbox)
                        elif TRACK_BASIC == 1:
                            state = predict_state(idx, state)
                            output1, bbox = update_output(idx, state, output1, prev_output)
                            state, bbox1 = update_state(idx, state, output1)
                            if DISPLAY:
                                image = cv2.imread(img_q)
                                show_frame_new(image, output1, idx, bbox)
                        elif DISPLAY == 1:
                            bbox = gen_bbox(prev_output, range(instance_num), True)
                            output1 = cv2.resize(output1, (prev_output.shape[1], prev_output.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
                            output_cluster_labels_n = cv2.resize(output_cluster_labels_n,
                                                                 (prev_output.shape[1], prev_output.shape[0]),
                                                                 interpolation=cv2.INTER_NEAREST)
                            if DISPLAY:
                                image = cv2.imread(img_q)
                                # show_frame_new(image,output1,idx,bbox)
                                output_cluster_labels_d =output_cluster_labels_n
                                indi = output_cluster_labels_d < cumsum_modes[1]
                                output_cluster_labels_d = output_cluster_labels_d - cumsum_modes[1] +1
                                output_cluster_labels_d[indi] = 0
                                show_clusters(name, output_cluster_labels_d, image, output1, idx, bbox)

                #########################################


                ####################
                ### Computing the accuracy:
                gt = query_label.numpy()[0]
                if (output.shape[0] != gt.shape[0]) or (output.shape[1] != gt.shape[1]):
                    output = cv2.resize(output1, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                if MEASURE:
                    M = np.zeros((nCls, nCls))
                    for i in range(M.shape[0]):
                        outg = output[gt == i]
                        for j in range(M.shape[1]):
                            M[i, j] = (outg == j).astype(np.float).sum()

                    if idx != 0:
                        for i in range(M.shape[0]):
                            den = (M[i, :].sum() + M[:, i].sum() - M[i, i])
                            if (gt == i).astype(np.float).sum() != 0:
                                mean_ious_seq[i] += M[i, i] / den
                                mean_ious_seq_len[i] += 1

                PALETTE = np.loadtxt('../davis-2017/data/palette.txt', dtype=np.uint8).reshape(-1, 3)
                if size[0] != 480 or size[1] != 854:
                    sz = size[:-1][::-1]
                    output = cv2.resize(output, sz, interpolation=cv2.INTER_NEAREST)
                if os.path.isdir('output_images_{}/{}'.format(VERSION, name[0])) != True:
                    os.mkdir('output_images_{}/{}'.format(VERSION, name[0]))
                imwrite_indexed('output_images_{}/{}/{:05}.png'.format(VERSION, name[0], idx), output.astype(np.uint8),
                                PALETTE)
                ##########################################

                prev_output = output


                ####################################################################################
                for l in range(nCls-1):
                    mask = (output == (l + 1)).astype(np.uint8)
                    if idx == 0:
                        mask = (gt == (l + 1)).astype(np.uint8)
                    if (mask.shape[0] != gt.shape[0]) or (mask.shape[1] != gt.shape[1]):
                        last_seen[l] = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST).copy()
                    else:
                        last_seen[l] = mask.copy()
                    if mask.sum() == 0:
                        # last_seen[l] = getBB(mask=last_seen[l], margin=60).astype(np.float)
                        last_seen[l] += 1
                ####################################################################################

                ####################################################################################
                ### ONLINE Adaptation:
                ####################################################################################
                if ONLINE_UPDATE and idx % Update_Frequency == 0 and idx != 0:
                    output_small = cv2.resize(output, (emb_size[2], emb_size[1]), interpolation=cv2.INTER_NEAREST)
                    N_CLUSTERS_ADAPT = min([MODES_ADDED] + [np.sum(output_small == l) for l in cls1])
                    if N_CLUSTERS_ADAPT <= 1:
                    	print(N_CLUSTERS_ADAPT)
                        continue
                    # if name[0] == 'bmx-trees' and idx == 70:
                    #     pdb.set_trace()
                    modes, mapr, class_labels_indcs = clustering_morph(embeddings[0].cpu().data.numpy(),
                                                                 (query_image[0].numpy().transpose(1, 2, 0) + IMG_MEAN).astype('uint8'),
                                                                 output_small, nCls, emb_size, nclusters=N_CLUSTERS_ADAPT,
                                                                 bg_factor=1, method='kmeans', pca_comps=None)
                    modes_mean = modes.mean(1)

                    number_of_modes_temp = np.zeros(nCls).astype(np.int16)
                    number_of_modes_temp[0] = N_CLUSTERS_ADAPT * 1
                    for i in range(1, nCls):
                        number_of_modes_temp[i] = N_CLUSTERS_ADAPT

                    nanidx = np.where(np.isnan(modes_mean))[0]
                    for x in nanidx:
                        clsnan = (x - number_of_modes_temp[0]) // N_CLUSTERS_ADAPT
                        cc = 0
                        if clsnan >= 0:
                            base = clsnan * N_CLUSTERS_ADAPT + number_of_modes_temp[0]
                        else:
                            base = 0
                        while np.isnan(modes_mean[base + cc]):
                            cc += 1
                        modes[x, :] = modes[base + cc, :]

                    c = -1
                    for i in range(nCls):
                        if not i in class_labels_indcs:
                            continue
                        base_modes = modes_base_list[i]
                        modes_correlation = np.matmul(modes, base_modes.transpose(1, 0))
                        c += 1
                        for j in range(c*N_CLUSTERS_ADAPT, (c+1)*N_CLUSTERS_ADAPT):
                            if modes_correlation[j, :].max() > Mode_Update_Simularity_Threshold:
                                base_modes = np.vstack([base_modes, modes[j, :]])
                                number_of_modes[i] += 1
                        modes_base_list[i] = base_modes

                    cumsum_modes = np.cumsum(number_of_modes)
                    cumsum_modes = np.hstack([0, cumsum_modes])
                ####################################################################################



            ####################
            ### Computing the accuracy:
            if MEASURE:
                mean_ious_seq = mean_ious_seq[1:] / mean_ious_seq_len[1:]
                mean_ious_seq = [round(mean_ious_seq[i], 3) for i in range(mean_ious_seq.shape[0])]
                print('{}: Total: mean-IoU = {}.\n'.format(name[0], mean_ious_seq))
                for iou in mean_ious_seq:
                    fin.write('{}\n'.format(iou))
            ####################

        #########################################
        ### Garbage Collection
        del model, embeddings, y
        gc.collect()
        torch.cuda.empty_cache()
        #########################################

        #########################################################
        if TRACKING:
            ############### TRACKING PART
            frame_cnt += nFrames
            #########################################################

    end = timeit.default_timer()
    print(end - start, 'seconds')
    fin.close()


def save_frame(bbox, th, do_pause, dir_name='', vis=True):
    result = prob_to_label(combine_prob(pred_prob[th]))
    result_show = np.dstack((colors[result, 0], colors[result, 1], colors[result, 2])).astype(np.uint8)
    temp = cv2.resize(frames, frame_0.shape[-2::-1]) * 0.3 + result_show * 0.7
    for i in range(instance_num):
        temp = cv2.rectangle(temp, (bbox[i, 0], bbox[i, 1]), (bbox[i, 2], bbox[i, 3]), (0, 255, 0), 5)
    cv2.imshow('Result', temp.astype(np.uint8))
    if do_pause:
        cv2.waitKey()
    else:
        cv2.waitKey(100)

    return


def show_frame(result, th, bbox):
    result_show = np.dstack((colors[result, 0], colors[result, 1], colors[result, 2])).astype(np.uint8)
    temp = cv2.resize(frames, frame_0.shape[-2::-1]) * 0.3 + result_show * 0.7
    for i in range(instance_num):
        temp = cv2.rectangle(temp, (bbox[i, 0], bbox[i, 1]), (bbox[i, 2], bbox[i, 3]), (0, 255, 0), 5)
    temp1 = cv2.resize(frames, frame_0.shape[-2::-1])
    showim = np.concatenate((temp1, temp), axis=1)
    cv2.imshow('result', showim.astype(np.uint8))
    cv2.waitKey(250)


def show_frame_new(supp_image, result, th, bbox):
    result_show = np.dstack((colors[result, 0], colors[result, 1], colors[result, 2])).astype(np.uint8)
    temp = supp_image * 0.3 + result_show * 0.7
    for i in range(instance_num):
        temp = cv2.rectangle(temp, (bbox[i, 0], bbox[i, 1]), (bbox[i, 2], bbox[i, 3]), (0, 255, 0), 5)
    showim = np.concatenate((supp_image, temp), axis=1)
    cv2.imshow('result', showim.astype(np.uint8))
    cv2.waitKey(25)
    # k = cv2.waitKey(0)
    # if k == ord('a'):    # Esc key to stop
    #     return


def show_clusters(name, output_cluster_labels_n, supp_image, result, th, bbox):

    # read_file = os.path.join('output_images_2017_64', str(name[0]), '%05d.png' % th)
    # im = Image.open(read_file)

    # annotation = np.atleast_3d(im)[...,0]
    # aph = np.array(im.getpalette()).reshape((-1,3))
    # #result = cv2.imread(read_file)
    # result=annotation

    # indic = result > 0
    # result_show1 = np.dstack((colors[result, 0], colors[result, 1], colors[result, 2])).astype(np.uint8)
    # #result_show = np.dstack((colors[output_cluster_labels_n, 0], colors[output_cluster_labels_n, 1],colors[output_cluster_labels_n, 2])).astype(np.uint8)

    # # supp_image_mod = supp_image
    # # supp_image_mod[indic] = supp_image_mod[indic]*0.3
    # # temp1 = supp_image_mod + result_show1 * 0.7

    # temp1 = supp_image * 0.3 + result_show1 * 0.7

    # # for i in range(instance_num):
    # #     temp = cv2.rectangle(temp, (bbox[i, 0], bbox[i, 1]), (bbox[i, 2], bbox[i, 3]), (0, 255, 0), 5)
    # #showim = np.concatenate((supp_image, temp, temp1), axis=1)
    # dest_dir = 'output_images_val/' + str(name[0])
    # if os.path.isdir(dest_dir) != True:
    #                 os.mkdir(dest_dir)
    # dest = os.path.join(dest_dir, '%05d.png' % th)
    # cv2.imwrite(dest, temp1)
    # # dest1 = dest_dir + '/' +  str(th) + 'o.jpg'
    # # cv2.imwrite(dest1, temp1)
    # # cv2.imshow('result', result.astype(np.uint8))
    # # # cv2.waitKey(25)
    # # k = cv2.waitKey(0)
    # # if k == ord('a'):    # Esc key to stop
    #     return

    indic = result > 0
    result_show1 = np.dstack((colors[result, 0], colors[result, 1], colors[result, 2])).astype(np.uint8)
    result_show = np.dstack((colors[output_cluster_labels_n, 0], colors[output_cluster_labels_n, 1],
                             colors[output_cluster_labels_n, 2])).astype(np.uint8)

    supp_image_mod = supp_image.copy()
    supp_image_mod[indic] = supp_image_mod[indic]*0.3

    temp = supp_image_mod + result_show * 0.7
    temp1 = supp_image_mod + result_show1 * 0.7
    # for i in range(instance_num):
    #     temp = cv2.rectangle(temp, (bbox[i, 0], bbox[i, 1]), (bbox[i, 2], bbox[i, 3]), (0, 255, 0), 5)
    showim = np.concatenate((supp_image, temp, temp1), axis=1)
    cv2.imshow('result', showim.astype(np.uint8))
    cv2.waitKey(25)


    # dest_dir = 'figures/' + str(th) + '.jpg'
    # cv2.imwrite(os.path.join('figures', '%05d.jpg' % (th)), showim)
    # k = cv2.waitKey(0)
    # if k == ord('a'):    # Esc key to stop
    #     return

    # plt.imshow(showim.astype(np.uint8))
    # plt.show()
    



def combine(output, warp_label, bbox, th):
    if COMBINE == 2 or TRACK_BASIC:
        output = cv2.resize(output, (warp_label.shape[1], warp_label.shape[0]), interpolation=cv2.INTER_NEAREST)
        new_output = np.zeros(warp_label.shape, dtype=int)
        for i in range(instance_num):
            new_output[bbox[i, 1]:bbox[i, 3], bbox[i, 0]:bbox[i, 2]] = output[bbox[i, 1]:bbox[i, 3],
                                                                       bbox[i, 0]:bbox[i, 2]]
        return new_output
    if COMBINE == 1:
        output = cv2.resize(output, (warp_label.shape[1], warp_label.shape[0]), interpolation=cv2.INTER_NEAREST)
        new_output = np.zeros(warp_label.shape, dtype=int)
        for i in range(instance_num):
            new_output[bbox[i, 1]:bbox[i, 3], bbox[i, 0]:bbox[i, 2]] = output[bbox[i, 1]:bbox[i, 3],
                                                                       bbox[i, 0]:bbox[i, 2]]
        pred_prob[th] = label_to_prob(new_output, instance_num)

        return new_output


if __name__ == '__main__':
    main()
