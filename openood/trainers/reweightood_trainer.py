import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config


class ReweightOODTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config

        if self.config.dataset.name == 'imagenet':
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

        self.temp = config.trainer.trainer_args.temp
        self.m_b = config.trainer.trainer_args.m_b
        self.c_b = config.trainer.trainer_args.c_b
        self.m_w = config.trainer.trainer_args.m_w
        self.c_w = config.trainer.trainer_args.c_w
        self.elu = nn.ELU()

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
            feature = self.net(data)
            feature = F.normalize(feature, dim=-1)
            mask = torch.eq(target.unsqueeze(1), target.unsqueeze(0))
            pos_mask = mask.triu(diagonal=1).view(-1)
            neg_mask = mask.logical_not().triu(diagonal=1).view(-1)
            sim = torch.matmul(feature, feature.t()).view(-1)
            pos, neg = sim[pos_mask], sim[neg_mask]
            weight_pos = torch.sigmoid(-(pos.detach() * self.m_w - self.c_w))
            weight_neg = torch.sigmoid(neg.detach() * self.m_b - self.c_b)
            pos = torch.logsumexp(-weight_pos * pos / self.temp, dim=0)
            neg = torch.logsumexp(weight_neg * neg / self.temp, dim=0)
            loss = self.elu(pos + neg)

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
