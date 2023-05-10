import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .base_trainer import BaseTrainer


class MCDTrainer(BaseTrainer):
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
        self.margin = config.trainer.margin
        self.epoch_ft = config.trainer.start_epoch_ft

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
            batch = next(train_dataiter)

            data = batch['data'].cuda()
            if epoch_idx < self.epoch_ft:
                logits1, logits2 = self.net(data, return_double=True)
                loss = F.cross_entropy(logits1, batch['label'].cuda()) \
                    + F.cross_entropy(logits2, batch['label'].cuda())

            elif self.train_unlabeled_loader and epoch_idx >= self.epoch_ft:
                try:
                    unlabeled_batch = next(unlabeled_dataiter)
                except StopIteration:
                    unlabeled_dataiter = iter(self.train_unlabeled_loader)
                    unlabeled_batch = next(unlabeled_dataiter)

                id_bs = data.size(0)

                unlabeled_data = unlabeled_batch['data'].cuda()
                all_data = torch.cat([data, unlabeled_data])
                logits1, logits2 = self.net(all_data, return_double=True)

                logits1_id, logits2_id = logits1[:id_bs], logits2[:id_bs]
                logits1_ood, logits2_ood = logits1[id_bs:], logits2[id_bs:]

                loss = F.cross_entropy(logits1_id, batch['label'].cuda()) \
                    + F.cross_entropy(logits2_id, batch['label'].cuda())

                ent = torch.mean(entropy(logits1_ood) - entropy(logits2_ood))
                loss_oe = F.relu(self.margin - ent)

                loss += self.lambda_oe * loss_oe

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


def entropy(logits):
    score = torch.softmax(logits, dim=0)
    logscore = torch.log(score)
    entropy = torch.sum(-score * logscore, dim=0)
    return entropy
