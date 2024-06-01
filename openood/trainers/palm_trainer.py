import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config


class PALMTrainer(nn.Module):
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:
        super(PALMTrainer, self).__init__()
        self.net = net
        self.train_loader = train_loader
        self.config = config

        if 'imagenet' in self.config.dataset.name:
            try:
                for name, p in self.net.backbone.named_parameters():
                    if not name.startswith('layer4'):
                        p.requires_grad = False
            except AttributeError:
                for name, p in self.net.module.backbone.named_parameters():
                    if not name.startswith('layer4'):
                        p.requires_grad = False

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
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

        self.num_classes = config.dataset.num_classes
        self.temp = config.trainer.trainer_args.temp
        self.nviews = config.trainer.trainer_args.nviews
        self.proto_m = config.trainer.trainer_args.proto_m
        self.cache_size = config.trainer.trainer_args.n_proto
        self.feat_dim = config.network.feat_dim
        self.epsilon = config.trainer.trainer_args.epsilon
        self.sinkhorn_iterations = config.trainer.trainer_args.sinkhorn_iter
        self.k = min(config.trainer.trainer_args.k, self.cache_size)
        self.lambda_pcon = config.trainer.trainer_args.lambda_pcon
        self.n_protos = self.cache_size * self.num_classes
        self.register_buffer('protos',
                             torch.rand(self.n_protos, self.feat_dim).cuda())
        self.protos = F.normalize(self.protos, dim=-1)

    def sinkhorn(self, features):
        out = torch.matmul(features, self.protos.detach().T)

        Q = torch.exp(out.detach() / self.epsilon).t(
        )  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if torch.isinf(sum_Q):
            self.protos = F.normalize(self.protos, dim=1, p=2)
            out = torch.matmul(features, self.ws(self.protos.detach()).T)
            Q = torch.exp(out.detach() / self.epsilon).t(
            )  # Q is K-by-B for consistency with notations from our paper
            sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            Q = F.normalize(Q, dim=1, p=1)
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q = F.normalize(Q, dim=0, p=1)
            Q /= B

        Q *= B
        return Q.t()

    def mle_loss(self, features, targets):
        # update prototypes by EMA
        anchor_labels = targets.contiguous().repeat(self.nviews).view(-1, 1)
        contrast_labels = torch.arange(self.num_classes).repeat(
            self.cache_size).view(-1, 1).cuda()
        mask = torch.eq(anchor_labels, contrast_labels.T).float().cuda()

        Q = self.sinkhorn(features)
        # topk
        if self.k > 0:
            update_mask = mask * Q
            _, topk_idx = torch.topk(update_mask, self.k, dim=1)
            topk_mask = torch.scatter(torch.zeros_like(update_mask), 1,
                                      topk_idx, 1).cuda()
            update_mask = F.normalize(F.normalize(topk_mask * update_mask,
                                                  dim=1,
                                                  p=1),
                                      dim=0,
                                      p=1)
        # original
        else:
            update_mask = F.normalize(F.normalize(mask * Q, dim=1, p=1),
                                      dim=0,
                                      p=1)
        update_features = torch.matmul(update_mask.T, features)

        protos = self.protos
        protos = self.proto_m * protos + (1 - self.proto_m) * update_features

        self.protos = F.normalize(protos, dim=1, p=2)

        Q = self.sinkhorn(features)

        proto_dis = torch.matmul(features, self.protos.detach().T)
        anchor_dot_contrast = torch.div(proto_dis, self.temp)
        logits = anchor_dot_contrast

        if self.k > 0:
            loss_mask = mask * Q
            _, topk_idx = torch.topk(update_mask, self.k, dim=1)
            topk_mask = torch.scatter(torch.zeros_like(update_mask), 1,
                                      topk_idx, 1).cuda()
            loss_mask = F.normalize(topk_mask * loss_mask, dim=1, p=1)
            masked_logits = loss_mask * logits
        else:
            masked_logits = F.normalize(Q * mask, dim=1, p=1) * logits

        pos = torch.sum(masked_logits, dim=1)
        neg = torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True))
        log_prob = pos - neg

        loss = -torch.mean(log_prob)
        return loss

    def proto_contra(self):

        protos = F.normalize(self.protos, dim=1)
        batch_size = self.num_classes

        proto_labels = torch.arange(self.num_classes).repeat(
            self.cache_size).view(-1, 1).cuda()
        mask = torch.eq(proto_labels, proto_labels.T).float().cuda()

        contrast_count = self.cache_size
        contrast_feature = protos

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), 0.5)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to('cuda'), 0)
        mask = mask * logits_mask

        pos = torch.sum(F.normalize(mask, dim=1, p=1) * logits, dim=1)
        neg = torch.log(torch.sum(logits_mask * torch.exp(logits), dim=1))
        log_prob = pos - neg

        # loss
        loss = -torch.mean(log_prob)
        return loss

    def train_epoch(self, epoch_idx):
        adjust_learning_rate(self.config, self.optimizer, epoch_idx - 1)

        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

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
            target = target.cuda()

            # forward
            loss = 0
            features = self.net(data)
            mle = self.mle_loss(features, target)
            loss += mle

            if self.lambda_pcon > 0:
                pcon = self.lambda_pcon * self.proto_contra()
                loss += pcon

            self.protos = self.protos.detach()

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
