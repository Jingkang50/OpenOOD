from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from .mds_tools import process_feature_type, reduce_feature_dim

to_np = lambda x: x.data.cpu().numpy()


def calculate_prob(feature, feature_mean, feature_prec, component_weight):
    prob_matrix = torch.cuda.FloatTensor([])
    for cluster_idx in range(len(feature_mean)):
        zero_f = feature - feature_mean[cluster_idx]
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, feature_prec),
                                   zero_f.t()).diag()
        prob_gau = torch.exp(term_gau)
        prob_matrix = torch.cat((prob_matrix, prob_gau.view(-1, 1)), 1)
    prob = torch.mm(prob_matrix, component_weight.view(-1, 1))
    return prob


def get_GMM_stat(train_loader, model, num_clusters_list, feature_type_list,
                 feature_process_list):
    feature_mean_list, feature_prec_list, component_weight_list, transform_matrix_list = [], [], [], []
    num_layer = len(num_clusters_list)
    for layer_idx in range(num_layer):
        num_clusters = num_clusters_list[layer_idx]
        feature_type = feature_type_list[layer_idx]
        feature_process = feature_process_list[layer_idx]
        feature_list_full, label_list_full = [], []
        with torch.no_grad():
            for batch in train_loader:
                data = batch['plain_data'].cuda()
                label = batch['label']
                _, feature_list = model(data, return_feature_list=True)
                feature_list = process_feature_type(feature_list[layer_idx],
                                                    feature_type)
                feature_list_full.extend(to_np(feature_list))
                label_list_full.extend(to_np(label))
        feature_list_full = np.array(feature_list_full)
        label_list_full = np.array(label_list_full)
        transform_matrix = reduce_feature_dim(feature_list_full,
                                              label_list_full, feature_process)
        new_feature_list = np.dot(feature_list_full, transform_matrix)
        gm = GaussianMixture(n_components=num_clusters,
                             random_state=0,
                             covariance_type='tied').fit(new_feature_list)
        feature_mean = gm.means_
        feature_prec = gm.precisions_
        component_weight = gm.weights_

        feature_mean_list.append(torch.FloatTensor(feature_mean).cuda())
        feature_prec_list.append(torch.FloatTensor(feature_prec).cuda())
        component_weight_list.append(
            torch.FloatTensor(component_weight).cuda())
        transform_matrix_list.append(
            torch.FloatTensor(transform_matrix).cuda())

    return feature_mean_list, feature_prec_list, component_weight_list, transform_matrix_list
