import time

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from ..losses import rew_ce, rew_sce
from .base_trainer import BaseTrainer


class UDGTrainer(BaseTrainer):
    def __init__(
        self,
        net: nn.Module,
        train_loader: DataLoader,
        train_unlabeled_loader: DataLoader,
        config: Config,
    ) -> None:
        super().__init__(net, train_loader, config)

        self.train_unlabeled_loader = train_unlabeled_loader

        self.num_clusters = config.trainer.num_clusters
        self.purity_ind_thresh = config.trainer.purity_ind_thresh
        self.purity_ood_thresh = config.trainer.purity_ood_thresh
        self.oe_enhance_ratio = config.trainer.oe_enhance_ratio
        self.lambda_oe = config.trainer.lambda_oe
        self.lambda_aux = config.trainer.lambda_aux

        # Init clustering algorithm
        self.k_means = KMeans(k=config.trainer.num_clusters,
                              pca_dim=config.trainer.pca_dim)

    def train_epoch(self, epoch_idx):
        self._run_clustering(epoch_idx)
        metrics = self._compute_loss(epoch_idx)

        return self.net, metrics

    def _compute_loss(self, epoch_idx):
        self.net.train()  # enter train mode

        loss_avg, loss_cls_avg, loss_oe_avg, loss_aux_avg = 0.0, 0.0, 0.0, 0.0
        train_dataiter = iter(self.train_loader)
        unlabeled_dataiter = iter(self.train_unlabeled_loader)
        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            try:
                unlabeled_batch = next(unlabeled_dataiter)
            except StopIteration:
                unlabeled_dataiter = iter(self.train_unlabeled_loader)
                unlabeled_batch = next(unlabeled_dataiter)
            data = batch['data'].cuda()
            unlabeled_data = unlabeled_batch['data'].cuda()

            # concat labeled and unlabeled data
            logits_cls, logits_aux = self.net(data, return_aux=True)
            logits_oe_cls, logits_oe_aux = self.net(unlabeled_data,
                                                    return_aux=True)

            # classification loss
            concat_logits_cls = torch.cat([logits_cls, logits_oe_cls])
            concat_label = torch.cat([
                batch['label'],
                unlabeled_batch['pseudo_label'].type_as(batch['label']),
            ])
            loss_cls = F.cross_entropy(
                concat_logits_cls[concat_label != -1],
                concat_label[concat_label != -1].cuda(),
            )
            # oe loss
            concat_softlabel = torch.cat(
                [batch['soft_label'], unlabeled_batch['pseudo_softlabel']])
            concat_conf = torch.cat(
                [batch['ood_conf'], unlabeled_batch['ood_conf']])
            loss_oe = rew_sce(
                concat_logits_cls[concat_label == -1],
                concat_softlabel[concat_label == -1].cuda(),
                concat_conf[concat_label == -1].cuda(),
            )
            # aux loss
            concat_logits_aux = torch.cat([logits_aux, logits_oe_aux])
            concat_cluster_id = torch.cat(
                [batch['cluster_id'], unlabeled_batch['cluster_id']])
            concat_cluster_reweight = torch.cat([
                batch['cluster_reweight'], unlabeled_batch['cluster_reweight']
            ])
            loss_aux = rew_ce(
                concat_logits_aux,
                concat_cluster_id.cuda(),
                concat_cluster_reweight.cuda(),
            )

            # loss addition
            loss = loss_cls + self.lambda_oe * loss_oe \
                + self.lambda_aux * loss_aux
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            with torch.no_grad():
                # exponential moving average, show smooth values
                loss_cls_avg = loss_cls_avg * 0.8 + float(loss_cls) * 0.2
                loss_oe_avg = loss_oe_avg * 0.8 + float(
                    self.lambda_oe * loss_oe) * 0.2
                loss_aux_avg = (loss_aux_avg * 0.8 +
                                float(self.lambda_aux * loss_aux) * 0.2)
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['train_cls_loss'] = loss_cls_avg
        metrics['train_oe_loss'] = loss_oe_avg
        metrics['train_aux_loss'] = loss_aux_avg
        metrics['loss'] = loss_avg

        return metrics

    def _run_clustering(self, epoch_idx):
        self.net.eval()

        start_time = time.time()
        # get data from train loader
        print('Clustering: starting gather training features...', flush=True)
        # gather train image feature
        train_idx_list, unlabeled_idx_list, feature_list, train_label_list = (
            [],
            [],
            [],
            [],
        )
        train_dataiter = iter(self.train_loader)
        for step in tqdm(range(1,
                               len(train_dataiter) + 1),
                         desc='Epoch {:03d} ID Clustering: '.format(epoch_idx),
                         position=0,
                         leave=True,
                         disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            index = batch['index']
            label = batch['label']
            # we use no augmented image for clustering
            data = batch['data_aux'].cuda()
            _, feature = self.net(data, return_feature=True)
            feature = feature.detach()
            # evaluation
            for idx in range(len(data)):
                train_idx_list.append(index[idx].tolist())
                train_label_list.append(label[idx].tolist())
                feature_list.append(feature[idx].cpu().tolist())
        num_train_data = len(feature_list)
        train_idx_list = np.array(train_idx_list, dtype=int)
        train_label_list = np.array(train_label_list, dtype=int)
        train_label_list = sort_array(train_label_list, train_idx_list)
        # in-distribution samples always have pseudo labels == actual labels
        self.train_loader.dataset.pseudo_label = train_label_list

        torch.cuda.empty_cache()

        # gather unlabeled image feature in order
        unlabeled_conf_list, unlabeled_pseudo_list = [], []
        unlabeled_dataiter = iter(self.train_unlabeled_loader)
        for step in tqdm(range(1,
                               len(unlabeled_dataiter) + 1),
                         desc='Epoch {:03d} OE Clustering: '.format(epoch_idx),
                         position=0,
                         leave=True,
                         disable=not comm.is_main_process()):
            batch = next(unlabeled_dataiter)
            index = batch['index']
            # we use no augmented image for clustering
            data = batch['data_aux'].cuda()
            logit, feature = self.net(data, return_feature=True)
            feature = feature.detach()
            logit = logit.detach()
            score = torch.softmax(logit, dim=1)
            conf, pseudo = torch.max(score, dim=1)
            # evaluation
            for idx in range(len(data)):
                unlabeled_idx_list.append(index[idx].tolist())
                feature_list.append(feature[idx].cpu().tolist())
                unlabeled_conf_list.append(conf[idx].cpu().tolist())
                unlabeled_pseudo_list.append(pseudo[idx].cpu().tolist())
        feature_list = np.array(feature_list)
        unlabeled_idx_list = np.array(unlabeled_idx_list, dtype=int)
        unlabeled_conf_list = np.array(unlabeled_conf_list)
        unlabeled_pseudo_list = np.array(unlabeled_pseudo_list)
        unlabeled_conf_list = sort_array(unlabeled_conf_list,
                                         unlabeled_idx_list)
        unlabeled_pseudo_list = sort_array(unlabeled_pseudo_list,
                                           unlabeled_idx_list)
        torch.cuda.empty_cache()

        print('\nAssigning Cluster Labels...', flush=True)
        cluster_id = self.k_means.cluster(feature_list)
        train_cluster_id = cluster_id[:num_train_data]
        unlabeled_cluster_id = cluster_id[num_train_data:]
        # assign cluster id to samples. Sorted by shuffle-recording index.
        train_cluster_id = sort_array(train_cluster_id, train_idx_list)
        unlabeled_cluster_id = sort_array(unlabeled_cluster_id,
                                          unlabeled_idx_list)
        self.train_loader.dataset.cluster_id = train_cluster_id
        self.train_unlabeled_loader.dataset.cluster_id = unlabeled_cluster_id
        cluster_id = np.concatenate([train_cluster_id, unlabeled_cluster_id])
        # reweighting based on samples in clusters
        cluster_stat = np.zeros(self.num_clusters)
        cluster_id_list, cluster_id_counts = np.unique(cluster_id,
                                                       return_counts=True)
        for cluster_idx, counts in zip(cluster_id_list, cluster_id_counts):
            cluster_stat[cluster_idx] = counts
        inv_class_freq = 1 / (cluster_stat + 1e-10)
        sample_weight = np.power(inv_class_freq, 0.5)
        sample_weight *= 1 / sample_weight.mean()
        sample_weight_list = np.array([sample_weight[i] for i in cluster_id])
        self.train_loader.dataset.cluster_reweight \
            = sample_weight_list[:num_train_data]
        self.train_unlabeled_loader.dataset.cluster_reweight \
            = sample_weight_list[num_train_data:]

        print('In-Distribution Filtering (with OOD Enhancement)...',
              flush=True)
        old_train_pseudo_label \
            = self.train_loader.dataset.pseudo_label
        old_unlabeled_pseudo_label \
            = self.train_unlabeled_loader.dataset.pseudo_label
        old_pseudo_label = np.append(old_train_pseudo_label,
                                     old_unlabeled_pseudo_label).astype(int)
        new_pseudo_label = (-1 * np.ones_like(old_pseudo_label)).astype(int)
        # process ood confidence for oe loss enhancement (ole)
        new_ood_conf = np.ones_like(old_pseudo_label).astype(float)

        total_num_to_filter = 0
        purity_ind_thresh = self.purity_ind_thresh
        purity_ood_thresh = self.purity_ood_thresh
        # pick out clusters with purity over threshold
        for cluster_idx in range(self.num_clusters):
            label_in_cluster, label_counts = np.unique(
                old_pseudo_label[cluster_id == cluster_idx],
                return_counts=True)
            cluster_size = len(old_pseudo_label[cluster_id == cluster_idx])
            purity = label_counts / cluster_size  # purity list for each label
            # idf
            if np.any(purity > purity_ind_thresh):
                majority_label = label_in_cluster[purity > purity_ind_thresh][
                    0]  # first element in the list
                new_pseudo_label[cluster_id == cluster_idx] = majority_label
                # this might also change some ID but nvm
                if majority_label > 0:  # ID cluster
                    num_to_filter = len(label_in_cluster == -1)
                    total_num_to_filter += num_to_filter
            # ole
            elif np.any(purity > purity_ood_thresh):
                majority_label = label_in_cluster[
                    purity > purity_ood_thresh][0]
                if majority_label == -1:
                    new_ood_conf[cluster_id ==
                                 cluster_idx] = self.oe_enhance_ratio
        print(f'{total_num_to_filter} sample(s) filtered!', flush=True)

        self.train_unlabeled_loader.dataset.pseudo_label = new_pseudo_label[
            num_train_data:]
        self.train_unlabeled_loader.dataset.ood_conf = new_ood_conf[
            num_train_data:]

        print('Randomize Auxiliary Head...', flush=True)
        if hasattr(self.net, 'fc_aux'):
            # reset auxiliary branch
            self.net.fc_aux.weight.data.normal_(mean=0.0, std=0.01)
            self.net.fc_aux.bias.data.zero_()
        else:
            # reset fc for unsupervised learning (baseline)
            self.net.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.net.fc.bias.data.zero_()

        print(
            '# Online Clustering Completed! Duration: {:.2f}s #'.format(
                time.time() - start_time),
            flush=True,
        )


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


def sort_array(old_array, index_array):
    sorted_array = np.ones_like(old_array)
    sorted_array[index_array] = old_array
    return sorted_array
