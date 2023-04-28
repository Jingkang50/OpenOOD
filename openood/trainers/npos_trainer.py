import faiss.contrib.torch_utils
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config


class NPOSTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config

        # a bunch of constants or hyperparams
        self.n_cls = config.dataset.num_classes
        self.sample_number = config.trainer.trainer_args.sample_number
        self.sample_from = config.trainer.trainer_args.sample_from
        try:
            self.penultimate_dim = net.backbone.feature_size
        except AttributeError:
            self.penultimate_dim = net.backbone.module.feature_size
        self.start_epoch_KNN = config.trainer.trainer_args.start_epoch_KNN
        self.K = config.trainer.trainer_args.K
        self.select = config.trainer.trainer_args.select
        self.cov_mat = config.trainer.trainer_args.cov_mat
        self.pick_nums = config.trainer.trainer_args.pick_nums
        self.w_disp = config.trainer.trainer_args.w_disp
        self.w_comp = config.trainer.trainer_args.w_comp
        self.loss_weight = config.trainer.trainer_args.loss_weight
        self.temp = config.trainer.trainer_args.temp
        self.ID_points_num = config.trainer.trainer_args.ID_points_num

        res = faiss.StandardGpuResources()
        self.KNN_index = faiss.GpuIndexFlatL2(res, self.penultimate_dim)

        self.number_dict = {}
        for i in range(self.n_cls):
            self.number_dict[i] = 0

        if self.config.num_gpus > 1:
            params = [{
                'params': net.module.backbone.parameters()
            }, {
                'params': net.module.head.parameters()
            }, {
                'params':
                net.module.mlp.parameters(),
                'lr':
                config.optimizer.lr * config.optimizer.mlp_decay_rate
            }]
        else:
            params = [{
                'params': net.backbone.parameters()
            }, {
                'params': net.head.parameters()
            }, {
                'params':
                net.mlp.parameters(),
                'lr':
                config.optimizer.lr * config.optimizer.mlp_decay_rate
            }]

        self.optimizer = torch.optim.SGD(
            params,
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        if config.dataset.train.batch_size \
                * config.num_gpus * config.num_machines > 256:
            config.optimizer.warm = True

        if config.optimizer.warm:
            self.warmup_from = 0.001
            self.warm_epochs = 10
            if config.optimizer.cosine:
                eta_min = config.optimizer.lr * \
                    (config.optimizer.lr_decay_rate**3)
                self.warmup_to = eta_min + (config.optimizer.lr - eta_min) * (
                    1 + math.cos(math.pi * self.warm_epochs /
                                 config.optimizer.num_epochs)) / 2
            else:
                self.warmup_to = config.optimizer.lr

        self.criterion_comp = CompLoss(self.n_cls,
                                       temperature=self.temp).cuda()
        # V2: EMA style prototypes
        self.criterion_disp = DispLoss(self.n_cls,
                                       config.network.feat_dim,
                                       config.trainer.trainer_args.proto_m,
                                       self.net,
                                       val_loader,
                                       temperature=self.temp).cuda()

    def train_epoch(self, epoch_idx):
        adjust_learning_rate(self.config, self.optimizer, epoch_idx - 1)

        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        data_dict = torch.zeros(self.n_cls, self.sample_number,
                                self.penultimate_dim).cuda()

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            warmup_learning_rate(self.config, self.warm_epochs,
                                 self.warmup_from,
                                 self.warmup_to, epoch_idx - 1, train_step,
                                 len(train_dataiter), self.optimizer)

            batch = next(train_dataiter)
            data = batch['data']
            target = batch['label']

            data = torch.cat([data[0], data[1]], dim=0).cuda()
            target = target.repeat(2).cuda()

            # forward
            penultimate = self.net.backbone(data)
            features = self.net.head(penultimate)

            sum_temp = 0
            for index in range(self.n_cls):
                sum_temp += self.number_dict[index]
            lr_reg_loss = torch.zeros(1).cuda()[0]

            if sum_temp == self.n_cls * self.sample_number \
                    and epoch_idx < self.start_epoch_KNN:
                # maintaining an ID data queue for each class.
                target_numpy = target.cpu().data.numpy()
                for index in range(len(target)):
                    dict_key = target_numpy[index]
                    data_dict[dict_key] = torch.cat(
                        (data_dict[dict_key][1:],
                         penultimate[index].detach().view(1, -1)), 0)
            elif sum_temp == self.n_cls * self.sample_number \
                    and epoch_idx >= self.start_epoch_KNN:
                target_numpy = target.cpu().data.numpy()
                for index in range(len(target)):
                    dict_key = target_numpy[index]
                    data_dict[dict_key] = torch.cat(
                        (data_dict[dict_key][1:],
                         penultimate[index].detach().view(1, -1)), 0)
                # Standard Gaussian distribution
                new_dis = MultivariateNormal(
                    torch.zeros(self.penultimate_dim).cuda(),
                    torch.eye(self.penultimate_dim).cuda())
                negative_samples = new_dis.rsample((self.sample_from, ))
                for index in range(self.n_cls):
                    ID = data_dict[index]
                    sample_point = generate_outliers(
                        ID,
                        input_index=self.KNN_index,
                        negative_samples=negative_samples,
                        ID_points_num=self.ID_points_num,
                        K=self.K,
                        select=self.select,
                        cov_mat=self.cov_mat,
                        sampling_ratio=1.0,
                        pic_nums=self.pick_nums,
                        depth=self.penultimate_dim)
                    if index == 0:
                        ood_samples = sample_point
                    else:
                        ood_samples = torch.cat((ood_samples, sample_point), 0)

                if len(ood_samples) != 0:
                    energy_score_for_fg = self.net.mlp(penultimate)
                    energy_score_for_bg = self.net.mlp(ood_samples)
                    input_for_lr = torch.cat(
                        (energy_score_for_fg, energy_score_for_bg),
                        0).squeeze()
                    labels_for_lr = torch.cat(
                        (torch.ones(len(energy_score_for_fg)).cuda(),
                         torch.zeros(len(energy_score_for_bg)).cuda()), -1)
                    criterion_BCE = torch.nn.BCEWithLogitsLoss()
                    lr_reg_loss = criterion_BCE(input_for_lr.view(-1),
                                                labels_for_lr)
            else:
                target_numpy = target.cpu().data.numpy()
                for index in range(len(target)):
                    dict_key = target_numpy[index]
                    if self.number_dict[dict_key] < self.sample_number:
                        data_dict[dict_key][self.number_dict[
                            dict_key]] = penultimate[index].detach()
                        self.number_dict[dict_key] += 1
            normed_features = F.normalize(features, dim=1)

            disp_loss = self.criterion_disp(normed_features, target)
            comp_loss = self.criterion_comp(normed_features,
                                            self.criterion_disp.prototypes,
                                            target)

            loss = self.w_disp * disp_loss + self.w_comp * comp_loss
            loss = self.loss_weight * lr_reg_loss + loss

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced


def adjust_learning_rate(config, optimizer, epoch):
    lr = config.optimizer.lr
    if config.optimizer.cosine:
        eta_min = lr * (config.optimizer.lr_decay_rate**3)
        lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / config.optimizer.num_epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(config.optimizer.lr_decay_epochs))
        if steps > 0:
            lr = lr * (config.optimizer.lr_decay_rate**steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(config, warm_epochs, warmup_from, warmup_to, epoch,
                         batch_id, total_batches, optimizer):
    if config.optimizer.warm and epoch <= warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (warm_epochs * total_batches)
        lr = warmup_from + p * (warmup_to - warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class CompLoss(nn.Module):
    def __init__(self, n_cls, temperature=0.07, base_temperature=0.07):
        super(CompLoss, self).__init__()
        self.n_cls = n_cls
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, prototypes, labels):
        device = torch.device('cuda')

        proxy_labels = torch.arange(0, self.n_cls).to(device)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, proxy_labels.T).float().to(device)

        # compute logits
        anchor_feature = features
        contrast_feature = prototypes / prototypes.norm(dim=-1, keepdim=True)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1)
        loss = -(self.temperature /
                 self.base_temperature) * mean_log_prob_pos.mean()
        return loss


class DispLoss(nn.Module):
    def __init__(self,
                 n_cls,
                 feat_dim,
                 proto_m,
                 model,
                 loader,
                 temperature=0.1,
                 base_temperature=0.1):
        super(DispLoss, self).__init__()
        self.n_cls = n_cls
        self.feat_dim = feat_dim
        self.proto_m = proto_m
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.register_buffer('prototypes',
                             torch.zeros(self.n_cls, self.feat_dim))
        self.model = model
        self.loader = loader
        self.init_class_prototypes()

    def forward(self, features, labels):
        prototypes = self.prototypes
        num_cls = self.n_cls
        for j in range(len(features)):
            prototypes[labels[j].item()] = F.normalize(
                prototypes[labels[j].item()] * self.proto_m + features[j] *
                (1 - self.proto_m),
                dim=0)
        self.prototypes = prototypes.detach()
        labels = torch.arange(0, num_cls).cuda()
        labels = labels.contiguous().view(-1, 1)

        mask = (1 - torch.eq(labels, labels.T).float()).cuda()

        logits = torch.div(torch.matmul(prototypes, prototypes.T),
                           self.temperature)

        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(num_cls).view(-1, 1).cuda(),
                                    0)
        mask = mask * logits_mask
        mean_prob_neg = torch.log(
            (mask * torch.exp(logits)).sum(1) / mask.sum(1))
        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
        loss = self.temperature / self.base_temperature * mean_prob_neg.mean()
        return loss

    def init_class_prototypes(self):
        """Initialize class prototypes."""
        self.model.eval()
        start = time.time()
        prototype_counts = [0] * self.n_cls
        with torch.no_grad():
            prototypes = torch.zeros(self.n_cls, self.feat_dim).cuda()
            for i, batch in enumerate(self.loader):
                input = batch['data']
                target = batch['label']
                input, target = input.cuda(), target.cuda()
                features = self.model(input)
                for j, feature in enumerate(features):
                    prototypes[target[j].item()] += feature
                    prototype_counts[target[j].item()] += 1
            for cls in range(self.n_cls):
                prototypes[cls] /= prototype_counts[cls]
            # measure elapsed time
            duration = time.time() - start
            print(f'Time to initialize prototypes: {duration:.3f}')
            prototypes = F.normalize(prototypes, dim=1)
            self.prototypes = prototypes


def generate_outliers(ID,
                      input_index,
                      negative_samples,
                      ID_points_num=2,
                      K=20,
                      select=1,
                      cov_mat=0.1,
                      sampling_ratio=1.0,
                      pic_nums=30,
                      depth=342):
    length = negative_samples.shape[0]
    data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)
    normed_data = ID / data_norm
    rand_ind = np.random.choice(normed_data.shape[0],
                                int(normed_data.shape[0] * sampling_ratio),
                                replace=False)
    index = input_index
    index.add(normed_data[rand_ind])
    minD_idx, k_th = KNN_dis_search_decrease(ID, index, K, select)
    minD_idx = minD_idx[np.random.choice(select, int(pic_nums), replace=False)]
    data_point_list = torch.cat(
        [ID[i:i + 1].repeat(length, 1) for i in minD_idx])
    negative_sample_cov = cov_mat * negative_samples.cuda().repeat(pic_nums, 1)
    negative_sample_list = negative_sample_cov + data_point_list
    point = KNN_dis_search_distance(negative_sample_list, index, K,
                                    ID_points_num, length, depth)

    index.reset()
    return point


def KNN_dis_search_distance(target,
                            index,
                            K=50,
                            num_points=10,
                            length=2000,
                            depth=342):
    '''
    data_point: Queue for searching k-th points
    target: the target of the search
    K
    '''
    # Normalize the features
    target_norm = torch.norm(target, p=2, dim=1, keepdim=True)
    normed_target = target / target_norm

    distance, output_index = index.search(normed_target, K)
    k_th_distance = distance[:, -1]
    k_th = k_th_distance.view(length, -1)
    # target_new = target.view(length, -1, depth)
    k_th_distance, minD_idx = torch.topk(k_th, num_points, dim=0)
    minD_idx = minD_idx.squeeze()
    point_list = []
    for i in range(minD_idx.shape[1]):
        point_list.append(i * length + minD_idx[:, i])
    return target[torch.cat(point_list)]


def KNN_dis_search_decrease(
    target,
    index,
    K=50,
    select=1,
):
    '''
    data_point: Queue for searching k-th points
    target: the target of the search
    K
    '''
    # Normalize the features
    target_norm = torch.norm(target, p=2, dim=1, keepdim=True)
    normed_target = target / target_norm

    distance, output_index = index.search(normed_target, K)
    k_th_distance = distance[:, -1]
    k_th_distance, minD_idx = torch.topk(k_th_distance, select)
    return minD_idx, k_th_distance
