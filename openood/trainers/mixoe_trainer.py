import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .base_trainer import BaseTrainer


class MixOETrainer(BaseTrainer):
    def __init__(
        self,
        net: nn.Module,
        train_loader: DataLoader,
        train_unlabeled_loader: DataLoader,
        config: Config,
    ) -> None:
        super().__init__(net, train_loader, config)
        self.train_unlabeled_loader = train_unlabeled_loader
        self.lambda_oe = config.trainer.lambda_oe
        self.alpha = config.trainer.alpha
        self.beta = config.trainer.beta
        self.mix_op = config.trainer.mix_op
        self.num_classes = config.dataset.num_classes
        self.criterion = SoftCE()

    def train_epoch(self, epoch_idx):
        self.net.train()  # enter train mode

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        if self.train_unlabeled_loader:
            unlabeled_dataiter = iter(self.train_unlabeled_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            # manually drop last batch to avoid batch size mismatch
            if train_step == len(train_dataiter):
                continue

            batch = next(train_dataiter)

            try:
                unlabeled_batch = next(unlabeled_dataiter)
            except StopIteration:
                unlabeled_dataiter = iter(self.train_unlabeled_loader)
                unlabeled_batch = next(unlabeled_dataiter)

            if len(unlabeled_batch['data']) < len(batch['data']):
                unlabeled_dataiter = iter(self.train_unlabeled_loader)
                unlabeled_batch = next(unlabeled_dataiter)

            x, y = batch['data'].cuda(), batch['label'].cuda()
            oe_x = unlabeled_batch['data'].cuda()
            bs = x.size(0)
            one_hot_y = torch.zeros(bs, self.num_classes).cuda()
            one_hot_y.scatter_(1, y.view(-1, 1), 1)

            # ID loss
            logits = self.net(x)
            id_loss = F.cross_entropy(logits, y)

            # MixOE loss
            # build mixed samples
            lam = np.random.beta(self.alpha, self.beta)

            if self.mix_op == 'cutmix':
                mixed_x = x.clone().detach()
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                           (x.size()[-1] * x.size()[-2]))
                # we empirically find that pasting outlier patch into ID data performs better
                # than pasting ID patch into outlier data
                mixed_x[:, :, bbx1:bbx2, bby1:bby2] = oe_x[:, :, bbx1:bbx2,
                                                           bby1:bby2]
            elif self.mix_op == 'mixup':
                mixed_x = lam * x + (1 - lam) * oe_x

            # construct soft labels and compute loss
            oe_y = torch.ones(oe_x.size(0),
                              self.num_classes).cuda() / self.num_classes
            soft_labels = lam * one_hot_y + (1 - lam) * oe_y
            mixed_loss = self.criterion(self.net(mixed_x), soft_labels)

            # Total loss
            loss = id_loss + self.lambda_oe * mixed_loss

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
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics


class SoftCE(nn.Module):
    def __init__(self, reduction='mean'):
        super(SoftCE, self).__init__()
        self.reduction = reduction

    def forward(self, logits, soft_targets):
        preds = logits.log_softmax(dim=-1)
        assert preds.shape == soft_targets.shape

        loss = torch.sum(-soft_targets * preds, dim=-1)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError("Reduction type '{:s}' is not supported!".format(
                self.reduction))


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
