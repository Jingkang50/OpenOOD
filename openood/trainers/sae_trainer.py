import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.losses import soft_cross_entropy
from openood.postprocessors.gmm_postprocessor import compute_single_GMM_score
from openood.postprocessors.mds_ensemble_postprocessor import (
    process_feature_type, reduce_feature_dim, tensor2list)
from openood.utils import Config

from .lr_scheduler import cosine_annealing
from .mixup_trainer import mixing, prepare_mixup


class SAETrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.trainer_args = self.config.trainer.trainer_args

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )

    @torch.no_grad()
    def setup(self):
        feature_all = None
        label_all = []
        # collect features
        for batch in tqdm(self.train_loader,
                          desc='Compute GMM Stats [Collecting]'):
            data = batch['data_aux'].cuda()
            label = batch['label']
            _, feature_list = self.net(data, return_feature_list=True)
            label_all.extend(tensor2list(label))
            feature_processed = process_feature_type(
                feature_list[0], self.trainer_args.feature_type)
            if isinstance(feature_all, type(None)):
                feature_all = tensor2list(feature_processed)
            else:
                feature_all.extend(tensor2list(feature_processed))
        label_all = np.array(label_all)

        # reduce feature dim and perform gmm estimation
        feature_all = np.array(feature_all)
        transform_matrix = reduce_feature_dim(feature_all, label_all,
                                              self.trainer_args.reduce_dim)
        feature_all = np.dot(feature_all, transform_matrix)
        # GMM estimation
        gm = GaussianMixture(n_components=self.trainer_args.num_clusters,
                             random_state=0,
                             covariance_type='tied').fit(feature_all)
        feature_mean = gm.means_
        feature_prec = gm.precisions_
        component_weight = gm.weights_

        self.feature_mean = torch.Tensor(feature_mean).cuda()
        self.feature_prec = torch.Tensor(feature_prec).cuda()
        self.component_weight = torch.Tensor(component_weight).cuda()
        self.transform_matrix = torch.Tensor(transform_matrix).cuda()

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            # mixup operation
            index, lam = prepare_mixup(batch, self.trainer_args.alpha)
            data_mix = mixing(batch['data'].cuda(), index, lam)
            soft_label_mix = mixing(batch['soft_label'].cuda(), index, lam)

            # classfication loss
            logits_cls = self.net(data)
            loss_clsstd = F.cross_entropy(logits_cls, target)  # standard cls
            logits_mix = self.net(data_mix)
            loss_clsmix = soft_cross_entropy(logits_mix, soft_label_mix)

            # source awareness enhancement
            prob_id = compute_single_GMM_score(self.net, data,
                                               self.feature_mean,
                                               self.feature_prec,
                                               self.component_weight,
                                               self.transform_matrix, 0,
                                               self.trainer_args.feature_type)
            prob_ood = compute_single_GMM_score(self.net, data_mix,
                                                self.feature_mean,
                                                self.feature_prec,
                                                self.component_weight,
                                                self.transform_matrix, 0,
                                                self.trainer_args.feature_type)
            loss_sae_id = 1 - torch.mean(prob_id)
            loss_sae_ood = torch.mean(prob_ood)

            # loss
            loss = self.trainer_args.loss_weight[0] * loss_clsstd \
                + self.trainer_args.loss_weight[1] * loss_clsmix \
                + self.trainer_args.loss_weight[2] * loss_sae_id \
                + self.trainer_args.loss_weight[3] * loss_sae_ood

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_avg

        return self.net, metrics
