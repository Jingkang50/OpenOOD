import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.utils import Config

from .lr_scheduler import cosine_annealing


class RTSTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config

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

            # forward
            logits_classifier, variance = self.net(data, return_var=True)
            epsilon = torch.randn_like(variance)
            temperature = torch.sum(
                variance * epsilon * epsilon, dim=1,
                keepdim=True) / (self.config.network.dof - 2)
            loss_kl = ((variance - torch.log(variance + 1e-8) - 1) *
                       0.5).mean()
            loss_head = F.cross_entropy(logits_classifier / temperature,
                                        target)
            loss = loss_head + self.config.network.kl_scale * loss_kl

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=self.net.parameters(),
                                     max_norm=2.5,
                                     norm_type=2)
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_avg

        return self.net, metrics
