from __future__ import absolute_import, division, print_function

import abc
import os

import faiss
import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from sklearn.random_projection import SparseRandomProjection
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z


def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list


class PatchcorePostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(PatchcorePostprocessor, self).__init__(config)
        self.config = config
        self.postprocessor_args = config.postprocessor.postprocessor_args
        self.n_neighbors = config.postprocessor.postprocessor_args.n_neighbors
        self.feature_mean, self.feature_prec = None, None
        self.alpha_list = None
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []
        self.features = []

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # step 1:
        self.model = net
        # on train start
        self.model.eval()  # to stop running_var move (maybe not critical)
        self.embedding_list = []

        if (self.config.network.load_cached_faiss):
            path = self.config.output_dir
            # load index
            if os.path.isfile(os.path.join(path, 'index.faiss')):
                self.index = faiss.read_index(os.path.join(
                    path, 'index.faiss'))
                if torch.cuda.is_available():
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.init_results_list()
                return

        # training step
        train_dataiter = iter(id_loader_dict['train'])

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            x = batch['data'].cuda()
            features = self.model.forward(x, return_feature=True)
            embeddings = []
            for feature in features:
                m = torch.nn.AvgPool2d(9, 1, 1)
                embeddings.append(m(feature))
            embedding = embedding_concat(embeddings[0], embeddings[1])
            self.embedding_list.extend(reshape_embedding(np.array(embedding)))

        # training end
        total_embeddings = np.array(self.embedding_list)

        # Random projection
        print('Random projection')
        self.randomprojector = SparseRandomProjection(
            n_components='auto',
            eps=0.9)  # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        print('Coreset Subsampling')
        selector = kCenterGreedy(total_embeddings, 0, 0)
        selected_idx = selector.select_batch(
            model=self.randomprojector,
            already_selected=[],
            N=int(total_embeddings.shape[0] *
                  self.postprocessor_args.coreset_sampling_ratio))
        self.embedding_coreset = total_embeddings[selected_idx]

        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        # faiss
        print('faiss indexing')
        self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        self.index.add(self.embedding_coreset)
        if not os.path.isdir(os.path.join('./results/patch/')):
            os.mkdir('./results/patch/')
        faiss.write_index(self.index,
                          os.path.join('./results/patch/', 'index.faiss'))

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []

    def postprocess(self, net: nn.Module, data):

        self.init_results_list()
        score_patch = []
        # extract embedding
        for x in data.split(1, dim=0):
            features = self.model.forward(x, return_feature=True)
            embeddings = []
            for feature in features:
                m = torch.nn.AvgPool2d(3, 1, 1)
                embeddings.append(m(feature))
            embedding_ = embedding_concat(embeddings[0], embeddings[1])
            embedding_test = np.array(reshape_embedding(np.array(embedding_)))
            score_patches, _ = self.index.search(embedding_test,
                                                 k=self.n_neighbors)

            score_patch.append(score_patches)

            N_b = score_patches[np.argmax(score_patches[:, 0])]
            w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
            score = w * max(score_patches[:, 0])  # Image-level score

            self.pred_list_img_lvl.append(score)

        pred = []
        for i in self.pred_list_img_lvl:
            # 6.3 is the trial value.
            if (i > 6.3):
                pred.append(torch.tensor(1))
            else:
                pred.append(torch.tensor(-1))
        conf = []
        for i in score_patch:
            conf.append(i)
        conf = torch.tensor(conf, dtype=torch.float32)
        conf = conf.cuda()

        pred_list_img_lvl = []

        for patchscore in np.concatenate([conf.cpu().tolist()]):
            N_b = patchscore[np.argmax(patchscore[:, 0])]
            w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
            score = w * max(patchscore[:, 0])  # Image-level score

            pred_list_img_lvl.append(score)

        if self.config.evaluator.name == 'patch':
            return pred, conf
        else:
            return pred, -1 * torch.tensor(pred_list_img_lvl).cuda()


# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Abstract class for sampling methods.

Provides interface to sampling methods that allow same signature for
select_batch.  Each subclass implements select_batch_ with the desired
signature for readability.
"""


class SamplingMethod(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, X, y, seed, **kwargs):
        self.X = X
        self.y = y
        self.seed = seed

    def flatten_X(self):
        shape = self.X.shape
        flat_X = self.X
        if len(shape) > 2:
            flat_X = np.reshape(self.X, (shape[0], np.product(shape[1:])))
        return flat_X

    @abc.abstractmethod
    def select_batch_(self):
        return

    def select_batch(self, **kwargs):
        return self.select_batch_(**kwargs)

    def to_dict(self):
        return None


# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Returns points that minimizes the maximum distance of any point to a center.

Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017

Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).

Can be extended to a robust k centers algorithm that ignores a certain number
of outlier datapoints.
Resulting centers are solution to multiple integer program.
"""


class kCenterGreedy(SamplingMethod):
    def __init__(self, X, y, seed, metric='euclidean'):
        self.X = X
        self.y = y
        self.flat_X = self.flatten_X()
        self.name = 'kcenter'
        self.features = self.flat_X
        self.metric = metric
        self.min_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []

    def update_distances(self,
                         cluster_centers,
                         only_new=True,
                         reset_dist=False):
        """Update min distances given cluster centers.

        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and
          update min_distances.
          rest_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [
                d for d in cluster_centers if d not in self.already_selected
            ]
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x, metric=self.metric)

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, model, already_selected, N, **kwargs):
        """Diversity promoting active learning method that greedily forms a
        batch to minimize the maximum distance to a cluster center among all
        unlabeled datapoints.

        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size

        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        try:
            # Assumes that the transform function takes in original data and
            # not flattened data.
            print('Getting transformed features...')
            self.features = model.transform(self.X)
            print('Calculating distances...')
            self.update_distances(already_selected,
                                  only_new=False,
                                  reset_dist=True)
        except:
            print('Using flat_X as features.')
            self.update_distances(already_selected,
                                  only_new=True,
                                  reset_dist=False)

        new_batch = []

        for _ in tqdm(range(N)):
            if self.already_selected is None:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f' %
              max(self.min_distances))

        self.already_selected = already_selected

        return new_batch
