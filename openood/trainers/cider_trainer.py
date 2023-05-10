import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config


class CIDERTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: Config) -> None:

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

        self.criterion_comp = CompLoss(
            config.dataset.num_classes,
            temperature=config.trainer.trainer_args.temp).cuda()
        # V2: EMA style prototypes
        self.criterion_dis = DisLoss(
            config.dataset.num_classes,
            config.network.feat_dim,
            config.trainer.trainer_args.proto_m,
            self.net,
            val_loader,
            temperature=config.trainer.trainer_args.temp).cuda()

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
            target = target.repeat(2).cuda()

            # forward
            features = self.net(data)
            dis_loss = self.criterion_dis(features, target)  # V2: EMA style
            comp_loss = self.criterion_comp(features,
                                            self.criterion_dis.prototypes,
                                            target)
            loss = self.config.trainer.trainer_args.w * comp_loss + dis_loss

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
    """Compactness Loss with class-conditional prototypes."""
    def __init__(self, n_cls, temperature=0.07, base_temperature=0.07):
        super(CompLoss, self).__init__()
        self.n_cls = n_cls
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, prototypes, labels):
        prototypes = F.normalize(prototypes, dim=1)
        proxy_labels = torch.arange(0, self.n_cls).cuda()
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, proxy_labels.T).float().cuda()  # bz, cls

        # compute logits
        feat_dot_prototype = torch.div(torch.matmul(features, prototypes.T),
                                       self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits = feat_dot_prototype - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1)

        # loss
        loss = -(self.temperature /
                 self.base_temperature) * mean_log_prob_pos.mean()
        return loss


class DisLoss(nn.Module):
    """Dispersion Loss with EMA prototypes."""
    def __init__(self,
                 n_cls,
                 feat_dim,
                 proto_m,
                 model,
                 loader,
                 temperature=0.1,
                 base_temperature=0.1):
        super(DisLoss, self).__init__()
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
