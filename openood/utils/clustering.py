import time

import faiss
import numpy as np


def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.

    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    return I.reshape(-1, )


class KMeans(object):
    def __init__(self, k, pca_dim):
        self.k = k
        self.pca_dim = pca_dim

    def cluster(self, data, verbose=True):
        """Performs k-means clustering.
        Args:
            x_data (np.array N * dim): data to cluster
        """
        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data, pca=self.pca_dim)

        if np.isnan(xb).any():
            row_sums = np.linalg.norm(data, axis=1)
            data_norm = data / row_sums[:, np.newaxis]
            if np.isnan(data_norm).any():
                I = run_kmeans(data_norm, self.k, verbose)
            else:
                I = run_kmeans(data, self.k, verbose)
        else:
            # cluster the data
            I = run_kmeans(xb, self.k, verbose)
        return I
