import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, MiniBatchKMeans, MeanShift
from sklearn.neighbors import kneighbors_graph
from skimage.segmentation import slic, felzenszwalb, quickshift
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl

import time
import copy
import pickle
import numpy as np
import torch
import pdb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn import mixture

import itertools
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

def plot_gmm_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

def clustering(embeddings, image, selected_pixels_list, emb_size, nclusters=2, bg_factor=1, method='kmeans', pca_comps=50):
    d, h, w = emb_size
    embeddings = embeddings.transpose(1, 0)
    class_cluster_modes = []
    class_labels_indcs = []
    mapr = -np.ones(h * w)
    xy = np.unravel_index(range(embeddings.shape[0]), emb_size[1:])
    # embeddings_tocluster = np.hstack([embeddings, xy[0][:, np.newaxis].astype(embeddings.dtype), xy[1][:, np.newaxis].astype(embeddings.dtype)])
    embeddings_tocluster = embeddings
    for l in range(len(selected_pixels_list)):
        sel = selected_pixels_list[l]
        embs = embeddings_tocluster[sel, :]

        if pca_comps is not None:
            pca = PCA(n_components=pca_comps)
            embs = pca.fit_transform(embs)

        if method == 'kmeans':
            # kmeans = KMeans(init='k-means++', n_clusters=nclusters, n_init=10)
            if l > 0:
                kmeans = MiniBatchKMeans(init='k-means++', n_clusters=nclusters, n_init=10, batch_size=500)
            else:  # Background is partitioned into more clusters
                kmeans = MiniBatchKMeans(init='k-means++', n_clusters=nclusters*bg_factor, n_init=10, batch_size=500)
            if embs.shape[0] < nclusters:
                pdb.set_trace()
                # factor = np.ceil(nclusters/embs.shape[0])
                # embs = embs[np.newaxis, :, :].repeat(factor, axis=0).reshape(-1, embs.shape[-1])
                # embs = embs[:nclusters, :]
                # sel = sel[np.newaxis, :].repeat(factor, axis=0).reshape(-1)
                # sel = sel[:nclusters]
            try:
                kmeans.fit(embs)
            except:
                pdb.set_trace()
            labels = kmeans.labels_

        elif method == 'gmm':
            dpgmm = mixture.BayesianGaussianMixture(n_components=nclusters, covariance_type='full').fit(embs)
            labels = dpgmm.predict(embs)
            # plot_results(pnts, labels, dpgmm.means_, dpgmm.covariances_, 1, 'Bayesian Gaussian Mixture with a Dirichlet process prior')
            # plt.show()

        elif method == 'agglomerative':
            connectivity = kneighbors_graph(embeddings_tocluster, n_neighbors=100, include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)
            average_linkage = AgglomerativeClustering(linkage="average", affinity="cityblock",
                n_clusters=nclusters, connectivity=connectivity)
            average_linkage.fit(embeddings_tocluster)
            labels = average_linkage.labels_

        elif method == 'dbscan':
            dbscan = DBSCAN(eps=.3)
            dbscan.fit(embeddings_tocluster)
            labels = dbscan.labels_

        elif method == 'tsne':
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_results = tsne.fit_transform(embs)
            # plot the result
            vis_x = tsne_results[:, 0]
            vis_y = tsne_results[:, 1]

            #labels = np.hstack([np.ones(100), 2 * np.ones(100)])
            labels = np.ones(vis_x.shape[0])
            ncls = np.unique(labels).shape[0]
            plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap("jet", ncls))
            plt.colorbar(ticks=range(ncls))
            plt.show()
            pdb.set_trace()
            continue

        else:
            raise NotImplementedError

        cluster_pos = []
        numclusters = labels.max() + 1
        for i in range(numclusters):
            cluster_indices = sel[labels == i]
            cluster_pos.append(cluster_indices)
            # if not cluster_indices.size:
            #     pdb.set_trace()
            class_cluster_modes.append(embeddings[cluster_indices, :].mean(axis=0))
            class_labels_indcs.append(l)

        ################################################
        ### Visualizing the cluster assignments
        base = mapr.max() + 1
        for i in range(numclusters):
            mapr[cluster_pos[i]] = base + i
        ################################################

    mapr = mapr.reshape((h,w))
    # plt.imshow(map), plt.show()
    class_cluster_modes = np.stack(class_cluster_modes)
    return class_cluster_modes, mapr, class_labels_indcs


import cv2
def clustering_morph(embeddings, image, label, nCls, emb_size, nclusters=2, bg_factor=1, method='kmeans', pca_comps=50):
    d, h, w = emb_size
    embeddings = embeddings.transpose(1, 0)
    class_cluster_modes = []
    class_labels_indcs = []
    mapr = -np.ones(h * w)
    xy = np.unravel_index(range(embeddings.shape[0]), emb_size[1:])
    # embeddings_tocluster = np.hstack([embeddings, xy[0][:, np.newaxis].astype(embeddings.dtype), xy[1][:, np.newaxis].astype(embeddings.dtype)])
    embeddings_tocluster = embeddings
    for l in range(nCls):

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        output_small = cv2.erode((label == l).astype(np.uint8), kernel, iterations=1)

        sel = np.where(output_small.reshape((1, -1)) == 1)[1]
        embs = embeddings_tocluster[sel, :]

        if pca_comps is not None:
            pca = PCA(n_components=pca_comps)
            embs = pca.fit_transform(embs)

        if method == 'kmeans':
            # kmeans = KMeans(init='k-means++', n_clusters=nclusters, n_init=10)
            if l > 0:
                kmeans = MiniBatchKMeans(init='k-means++', n_clusters=nclusters, n_init=10, batch_size=500)
            else:  # Background is partitioned into more clusters
                kmeans = MiniBatchKMeans(init='k-means++', n_clusters=nclusters*bg_factor, n_init=10, batch_size=500)
            try:
                kmeans.fit(embs)
            except:
                continue
            labels = kmeans.labels_

        elif method == 'gmm':
            dpgmm = mixture.BayesianGaussianMixture(n_components=nclusters, covariance_type='full').fit(embs)
            labels = dpgmm.predict(embs)
            # plot_results(pnts, labels, dpgmm.means_, dpgmm.covariances_, 1, 'Bayesian Gaussian Mixture with a Dirichlet process prior')
            # plt.show()

        elif method == 'agglomerative':
            connectivity = kneighbors_graph(embeddings_tocluster, n_neighbors=100, include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)
            average_linkage = AgglomerativeClustering(linkage="average", affinity="cityblock",
                n_clusters=nclusters, connectivity=connectivity)
            average_linkage.fit(embeddings_tocluster)
            labels = average_linkage.labels_

        elif method == 'dbscan':
            dbscan = DBSCAN(eps=.3)
            dbscan.fit(embeddings_tocluster)
            labels = dbscan.labels_

        elif method == 'tsne':
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_results = tsne.fit_transform(embs)
            # plot the result
            vis_x = tsne_results[:, 0]
            vis_y = tsne_results[:, 1]

            #labels = np.hstack([np.ones(100), 2 * np.ones(100)])
            labels = np.ones(vis_x.shape[0])
            ncls = np.unique(labels).shape[0]
            plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap("jet", ncls))
            plt.colorbar(ticks=range(ncls))
            plt.show()
            pdb.set_trace()
            continue

        else:
            raise NotImplementedError

        cluster_pos = []
        numclusters = labels.max() + 1
        for i in range(numclusters):
            cluster_indices = sel[labels == i]
            cluster_pos.append(cluster_indices)
            # if not cluster_indices.size:
            #     pdb.set_trace()
            class_cluster_modes.append(embeddings[cluster_indices, :].mean(axis=0))

        class_labels_indcs.append(l)

        ################################################
        ### Visualizing the cluster assignments
        base = mapr.max() + 1
        for i in range(numclusters):
            mapr[cluster_pos[i]] = base + i
        ################################################

    mapr = mapr.reshape((h,w))
    # plt.imshow(map), plt.show()
    class_cluster_modes = np.stack(class_cluster_modes)
    return class_cluster_modes, mapr, class_labels_indcs


def segmentation(image, nclusters=2, method='slic'):
    if method == 'slic':
        labels = slic(image, n_segments=nclusters, compactness=5, sigma=2.5)
    elif method == 'fz':
        labels = felzenszwalb(image, scale=1500, sigma=0.9, min_size=2500)
    elif method == 'qs':
        labels = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)
    else:
        raise NotImplementedError
    return labels