import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
import time
import cv2


def label_transfer(scores, labels_1, images_2, emb_size, picked_points, flow_2=None, avg_k_max=None, crf_niter=0):
	n, h, w = emb_size
	scores = nn.Sigmoid()(scores)
	cls1 = torch.from_numpy(np.unique(labels_1[labels_1>=0].numpy()))
	scores_picked = torch.zeros((cls1.size()[0], h * w))
	pidx = 0
	lbl_map = []
	if avg_k_max is None:
		for i, l in enumerate(cls1):
			lbl_map.append(l)
			scores_picked[i, :] = scores.data.cpu()[0, 0, pidx:(pidx + len(picked_points[i])), :].sum(dim=0) / len(picked_points[i])
			pidx += len(picked_points[i])
	elif avg_k_max == 1:
		for i, l in enumerate(cls1):
			lbl_map.append(l)
			scores_picked[i, :] = scores.data.cpu()[0, 0, pidx:(pidx + len(picked_points[i])), :].max(dim=0)[0]
			pidx += len(picked_points[i])
	else:
		for i, l in enumerate(cls1):
			lbl_map.append(l)
			K = avg_k_max if avg_k_max < len(picked_points[i]) else len(picked_points[i])
			scores_picked[i, :] = scores.data.cpu()[0, 0, pidx:(pidx + len(picked_points[i])), :].sort(dim=0)[0][-K:, :].sum(dim=0) / K
			pidx += len(picked_points[i])

	output = scores_picked / scores_picked.sum(dim=0)
	output = output.view(-1, h, w).unsqueeze(0)
	N, H, W = labels_1.size()
	output = F.upsample(output, (H, W), mode='bilinear')
	output_un = output.data.numpy()[0, 0]

	output_forcrf = output[0].data.numpy()
	# pdb.set_trace()

	#cur_time = time.time() - start_time
	#print('Score to prediction in {} seconds.'.format(cur_time))
	#start_time = time.time()

	start_time = time.time()
	##################################
	### Applying CRF:
	if crf_niter > 0:
		d = dcrf.DenseCRF2D(images_2.size()[3], images_2.size()[2], cls1.size()[0])
		unary = unary_from_softmax(output_forcrf)

		d.setUnaryEnergy(unary)
		img = np.ascontiguousarray(images_2.numpy()[0].transpose(1, 2, 0), dtype=np.uint8)
		d.addPairwiseGaussian(sxy=(3, 3), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
		d.addPairwiseBilateral(sxy=(80, 80), srgb=(10, 10, 10), rgbim=img, compat=10, kernel=dcrf.DIAG_KERNEL,
							   normalization=dcrf.NORMALIZE_SYMMETRIC)
		### Flow feature for CRF:
		if flow_2 is not None:
			fl = flow_2[0].numpy()
			fl[:, :, 0] -= fl[:, :, 0].min()
			fl[:, :, 1] -= fl[:, :, 1].min()
			fl = fl.mean(-1)
			fl = cv2.normalize(fl, None, 0, 255, cv2.NORM_MINMAX)
			flow_img = np.ascontiguousarray(fl, dtype=np.uint8)
			flow_potential = create_pairwise_bilateral(sdims=(80, 80), schan=(10), img=flow_img, chdim=-1).astype(np.float32)
			d.addPairwiseEnergy(flow_potential, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
		###
		Q = d.inference(crf_niter)
		MAP = np.argmax(Q, axis=0)
		output = MAP.reshape((img.shape[0], img.shape[1]))
	else:
		output = np.argmax(output_forcrf, axis=0)

	output_raw = output.copy()
	for i, l in enumerate(lbl_map):
		output[output_raw==i] = l
	return output, output_un


def label_transfer_end2end(scores, labels_1, images, emb_size, picked_points, flow_2=None, crf_niter=0):
	N, C, H, W = images.size()
	ncls = np.unique(labels_1[labels_1>=0].numpy()).shape[0]

	#scores = nn.Sigmoid()(scores)

	scores_picked = []
	pidx = 0
	for i in range(ncls):
		scores_picked.append(scores[0, :, pidx:(pidx + len(picked_points[i])), :].sum(dim=1) / len(picked_points[i]))
		pidx += len(picked_points[i])
	scores_end2end = torch.stack(scores_picked, dim=1)
	scores_end2end = scores_end2end.view(1, 2, emb_size[1], emb_size[2])
	scores_end2end = F.upsample(scores_end2end, (H, W), mode='bilinear')
	output = F.upsample(scores_end2end, (H, W), mode='bilinear')
	output_un = output.cpu().data.numpy()[0, 0]

	output_forcrf = output[0].cpu().data.numpy()
	# pdb.set_trace()

	#cur_time = time.time() - start_time
	#print('Score to prediction in {} seconds.'.format(cur_time))
	#start_time = time.time()

	start_time = time.time()
	##################################
	### Applying CRF:
	if crf_niter > 0:
		d = dcrf.DenseCRF2D(images.size()[3], images.size()[2], ncls)
		unary = unary_from_softmax(output_forcrf)

		d.setUnaryEnergy(unary)
		img = np.ascontiguousarray(images.numpy()[0].transpose(1, 2, 0), dtype=np.uint8)
		d.addPairwiseGaussian(sxy=(3, 3), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
		d.addPairwiseBilateral(sxy=(80, 80), srgb=(10, 10, 10), rgbim=img, compat=10, kernel=dcrf.DIAG_KERNEL,
							   normalization=dcrf.NORMALIZE_SYMMETRIC)
		### Flow feature for CRF:
		if flow_2 is not None:
			fl = flow_2[0].numpy()
			fl[:, :, 0] -= fl[:, :, 0].min()
			fl[:, :, 1] -= fl[:, :, 1].min()
			fl = fl.mean(-1)
			fl = cv2.normalize(fl, None, 0, 255, cv2.NORM_MINMAX)
			flow_img = np.ascontiguousarray(fl, dtype=np.uint8)
			flow_potential = create_pairwise_bilateral(sdims=(80, 80), schan=(10), img=flow_img, chdim=-1).astype(np.float32)
			d.addPairwiseEnergy(flow_potential, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
		###
		Q = d.inference(crf_niter)
		MAP = np.argmax(Q, axis=0)
		output = MAP.reshape((img.shape[0], img.shape[1]))
	else:
		output = np.argmax(output_forcrf, axis=0)

	return output, output_un


def apply_crf(probs, images, IMG_MEAN, flow_2=None, crf_niter=0):
	N, ncls, H, W = probs.shape

	probs = probs[0]
	# pdb.set_trace()

	#cur_time = time.time() - start_time
	#print('Score to prediction in {} seconds.'.format(cur_time))
	#start_time = time.time()

	start_time = time.time()
	##################################
	### Applying CRF:
	if crf_niter > 0:
		d = dcrf.DenseCRF2D(images.size()[3], images.size()[2], ncls)
		unary = unary_from_softmax(probs)

		d.setUnaryEnergy(unary)
		img = np.ascontiguousarray(images.numpy()[0].transpose(1, 2, 0) + IMG_MEAN, dtype=np.uint8)
		d.addPairwiseGaussian(sxy=(3, 3), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
		d.addPairwiseBilateral(sxy=(80, 80), srgb=(10, 10, 10), rgbim=img, compat=10, kernel=dcrf.DIAG_KERNEL,
							   normalization=dcrf.NORMALIZE_SYMMETRIC)
		### Flow feature for CRF:
		if flow_2 is not None:
			fl = flow_2
			fl[:, :, 0] -= fl[:, :, 0].min()
			fl[:, :, 1] -= fl[:, :, 1].min()
			fl = fl.mean(-1)
			fl = cv2.resize(fl, (W, H), interpolation=cv2.INTER_CUBIC)
			fl = cv2.normalize(fl, None, 0, 255, cv2.NORM_MINMAX)
			flow_img = np.ascontiguousarray(fl, dtype=np.uint8)
			flow_potential = create_pairwise_bilateral(sdims=(80, 80), schan=(10), img=flow_img, chdim=-1).astype(np.float32)
			d.addPairwiseEnergy(flow_potential, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
		###
		Q = d.inference(crf_niter)
		MAP = np.argmax(Q, axis=0)
		output = MAP.reshape((img.shape[0], img.shape[1]))
	else:
		output = np.argmax(probs, axis=0)

	return output
