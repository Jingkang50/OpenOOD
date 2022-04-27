import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.utils import Config

from .lr_scheduler import cosine_annealing


class CutPasteTrainer:
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

        embeds = []

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            data = torch.cat(batch['data'], 0)
            data = data.cuda()
            y = torch.arange(2)
            y = y.repeat_interleave(len(batch['data'][0]))
            y = y.cuda()
            # forward
            embed, logits_classifier = self.net(data)
            loss = F.cross_entropy(logits_classifier, y)

            embeds.append(embed.cuda())

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        embeds = torch.cat(embeds)
        embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_avg

        return self.net, metrics
