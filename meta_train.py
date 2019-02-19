import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import pickle
import cv2
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
import pdb
import timeit
from sklearn.metrics import confusion_matrix
from get_params import *
from loss_func import binary_cross_entropy2d, cross_entropy2d
from label_transfer import label_transfer
from datetime import datetime
from utils import read_image_label
from tqdm import tqdm
from deeplab.clustering import clustering
import copy

start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

BATCH_SIZE = 1
DATA_DIRECTORY = '../davis-2017/data/DAVIS'
VERSION = '2017'
DATA_LIST_PATH = '../davis-2017/data/DAVIS/ImageSets/2017/train_videos_{}.txt'.format(VERSION)
IGNORE_LABEL = 255
INPUT_SIZE = '240,427'
LEARNING_RATE = 2.5e-5
MOMENTUM = 0.9
NUM_CLASSES = 2
NUM_STEPS = 150001
POWER = 0.9
# RESTORE_FROM = './dataset/MS_DeepLab_resnet_pretrained_COCO_init.pth'
# RESTORE_FROM = './snapshots/Pascal_train_similarityloss_bg_50000.pth'
RESTORE_FROM = './snapshots/DAVIS_{}_train_on_pascal_1000.pth'.format(VERSION)
# RESTORE_FROM = './snapshots/DAVIS_{}_prototypical_MODES_train_max_62000.pth'.format(VERSION)
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

###########################################################################################
PROTOTYPICAL_SINGLE_MODE = False  # If False, then it uses NUMBER_OF_CLUSTERS modes
NUMBER_OF_CLUSTERS = 10
BG_Factor = 1
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
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()


args = get_arguments()


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


def main():
    """Create the model and start the training."""

    torch.manual_seed(1364)
    torch.cuda.manual_seed_all(1364)
    np.random.seed(1364)

    H, W = map(int, args.input_size.split(','))
    input_size = (H, W)

    cudnn.enabled = True
    gpu = args.gpu
    device = torch.device('cuda:{}'.format(gpu))

    # Create network.
    emb_size = (128, H, W)
    model = ResNetUpsampled_Deeplab(emb_size)
    # For a small batch size, it is better to keep
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model.
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    if RESTORE_FROM == './dataset/MS_DeepLab_resnet_pretrained_COCO_init.pth':
        saved_state_dict = torch.load(args.restore_from)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if i_parts[0]=='Scale':
                i_parts = i_parts[1:]
            if args.num_classes == 21 or not i_parts[0] == 'layer5':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
        model.load_state_dict(new_params)
    else:
        saved_state_dict = torch.load(args.restore_from, map_location={'cuda:3' : 'cuda:0', 'cuda:1' : 'cuda:0', 'cuda:2' : 'cuda:0'})
        model.load_state_dict(saved_state_dict)

    # model.float()
    model.eval() # use_global_stats = True
    # model.train()
    model.to(device)

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(DavisSiameseMAMLSet(args.data_dir, args.data_list,
                                                    max_iters=args.num_steps, all_frames=True, crop_size=input_size, mean=IMG_MEAN),
                                                    batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)

    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.learning_rate},
                           {'params': get_10x_lr_params(model), 'lr': args.learning_rate}],
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    counter = 0
    video_loss_neg = 0
    video_loss = 0
    fin = open('logs/training_' + str(datetime.now().time()), 'w+')

    for i_iter, batch in enumerate(trainloader):
        support_x, support_y, query_x, query_y, name = batch
        # if name[0] not in ['bus', 'bear']:
        #     continue

        # if i_iter <= 160:
        #     if i_iter % 100 == 0: print(i_iter)
        #     continue
        print(name)

        #########################################################
        ### Reading the first frame:
        ###################
        img_s = support_x[0]
        lab_s = support_y[0]
        supp_image, supp_label, _ = read_image_label(img_s, lab_s, mean=IMG_MEAN, mirror=False, scale=False, rotate=False)
        if VERSION=='2016':
            supp_label[supp_label > 0] = 1  # Single object segmentation
        supp_label = cv2.resize(supp_label[0], (emb_size[2], emb_size[1]), interpolation=cv2.INTER_NEAREST)[np.newaxis, :, :]
        supp_image = torch.from_numpy(supp_image)
        supp_label = torch.from_numpy(supp_label)
        # supp_label[supp_label > 0] = 1 #Single object segmentation; we ignore other labels and reduce everything to one.
        supp_label = supp_label.long()
        cls1 = np.unique(supp_label[supp_label >= 0].numpy())
        nCls = cls1.shape[0]
        #########################################################

        if not PROTOTYPICAL_SINGLE_MODE:
            #########################################################
            ### Computing the instance prototypes in the first frame:
            ###################
            embedding_reference = model(supp_image.to(device))
            supp_label_resh = supp_label.view(1, -1)
            embedding_reference_resh = embedding_reference.view(emb_size[0], -1)
            selected_pixels_list = [np.where(supp_label_resh.numpy()==l)[1] for l in cls1]
            ## If number of pixels in the object is less than NUMBER_OF_CLUSTERS, N_CLUSTERS should be the lower value.
            N_CLUSTERS = min([NUMBER_OF_CLUSTERS] + [np.sum(supp_label.numpy()[0] == l) for l in cls1])
            modes_1st, mapr, class_labels_indcs = clustering(embedding_reference_resh.cpu().data.numpy(),
                                                         (supp_image[0].numpy().transpose(1, 2, 0) + IMG_MEAN).astype('uint8'),
                                                         selected_pixels_list, emb_size, nclusters=N_CLUSTERS,
                                                         bg_factor=BG_Factor, method='kmeans', pca_comps=None)

            number_of_modes = np.zeros(nCls).astype(np.int16)
            number_of_modes[0] = N_CLUSTERS * BG_Factor
            for i in range(1, nCls):
                number_of_modes[i] = N_CLUSTERS

            cumsum_modes = np.cumsum(number_of_modes)
            cumsum_modes = np.hstack([0, cumsum_modes])
            #########################################################

        #########################################################
        ### Compute a total loss for the rest of the video frames:
        ###################
        nFrames = len(query_x)
        nCls = cls1.shape[0]
        mean_ious_seq = np.zeros(nCls)
        mean_ious_seq_len = np.zeros(nCls)

        optimizer.zero_grad()

        #####################################
        Num_Q_Frames_per_Episode = 2  # Number of query frames to use for meta-learning
        #####################################

        picked_query_frames = np.random.choice(range(0, nFrames), Num_Q_Frames_per_Episode, replace=False)
        # print(picked_query_frames)
        for idx in tqdm(picked_query_frames, total=Num_Q_Frames_per_Episode, desc='Computing the loss for the video:{}'.format(name[0]), ncols=120, leave=False):


            img_q = query_x[idx][0]
            lab_q = query_y[idx][0]
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

            embedding_reference = model(supp_image.to(device))
            embedding_reference_resh = embedding_reference.view(emb_size[0], -1)

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
                loss = F.cross_entropy(y, query_label_resh, ignore_index=-1, size_average=True) / float(Num_Q_Frames_per_Episode)
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
                    for c in range(cumsum_modes[l], cumsum_modes[l+1]):
                        indices = (cluster_map == c)
                        idx = np.where(indices.reshape(-1) > 0)[0]
                        if indices.size > 0:
                            cc += 1
                            cluster_embeddings = embedding_reference_resh[:, idx]
                            cluster_mean_embedding = cluster_embeddings.mean(dim=1)
                            modes.append(cluster_mean_embedding)
                    ### If number of computed clusters within an object is less than N_CLUSTERS,
                    ### then fill/replicate the rest of N_CLUSTER embeddings with previous clusters of that object:
                    for j in range(number_of_modes[l] - cc):
                        modes.append(modes[cumsum_modes[l] + j])
                modes = torch.stack(modes, dim=0).unsqueeze(0).to(device)

                DO_NORMALIZATION = 0
                if DO_NORMALIZATION:
                    modes = modes / torch.norm(modes, dim=-1).view(modes.size()[0], modes.size()[1], 1)
                    embeddings = embeddings / torch.norm(embeddings, dim=1).view(embeddings.size()[0], 1, embeddings.size()[-1])

                ### Computing the similarity
                y = torch.matmul(modes, embeddings)
                y = F.softmax(y, dim=1)
                y = y[0].permute(1, 0)
                ### Extracting the maximum probability cluster of each class (To be used for evaluation, not in the loss):
                output_probs = torch.stack([y[:, cumsum_modes[l]:cumsum_modes[l+1]].max(1)[0] for l in cls1], dim=-1).cpu().data.numpy().copy()
                y_parts = []
                for l in cls1:
                    y_part = y[:, cumsum_modes[l]:cumsum_modes[l+1]]
                    ### Taking argmax of the probabilities of the clusters of the correct class for each sample and
                    ### selecting the one with maximum score and computing the loss only for that cluster in each class:
                    ymaxidx = y_part.max(1)[1].unsqueeze(1).long()
                    ymaxidx_oh = 0 * torch.LongTensor(ymaxidx.size()[0], number_of_modes[l]).to(device)
                    ymaxidx_oh.scatter_(1, ymaxidx, 1)
                    y_part = y_part[ymaxidx_oh.byte()]
                    y_parts.append(y_part)
                y = torch.stack(y_parts, dim=1)
                y = y / (y.sum(1).unsqueeze(1))
                loss = F.cross_entropy(y, query_label_resh, ignore_index=-1, size_average=True) / float(Num_Q_Frames_per_Episode)

                # '''
                ####################################
                ### Computing the negative loss:
                ql_oh = 0 * torch.LongTensor(query_label_resh.size()[0], cls1.shape[0]).to(device)
                ql_oh.scatter_(1, query_label_resh.unsqueeze(1), 1)
                ### Negative loss is computed only for incorrect classes: (1-ql_oh)
                loss_neg = -(1-y).clamp(min=1e-12).log() * (1-ql_oh).float()
                loss_neg = loss_neg.mean() / float(Num_Q_Frames_per_Episode)

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

        counter += 1
        if i_iter % 1 == 0:
            print('iter = {} of {}, mean-loss = {:.04f}, mean-loss-neg = {:.04f}, mean-iou = {:.04f}'.format
                  (i_iter, args.num_steps, video_loss / float(counter), video_loss_neg / float(counter), np.mean(mean_ious_seq[1:] / mean_ious_seq_len[1:])))
            fin.write('iter = {} of {}, mean-loss = {:.04f}, mean-loss-neg = {:.04f}, mean-iou = {:.04f}, video = {}\n'.format
                  (i_iter, args.num_steps, video_loss / float(counter), video_loss_neg / float(counter), np.mean(mean_ious_seq[1:] / mean_ious_seq_len[1:]), name[0]))
            counter = 0
            video_loss_neg = 0
            video_loss = 0

        ##################################
        ####### Saving an output based on the model learned so far
        if i_iter % 100 == 0 and i_iter != 0:
            print('Saving sample output ...')
        ##################################

        ##################################
        ####### Saving snapshots
        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'DAVIS_{}_prototypical_MODES_train_max_{}.pth'.format(VERSION, i_iter)))
        ##################################


    end = timeit.default_timer()
    print(end - start, 'seconds')
    fin.close()


if __name__ == '__main__':
    main()
