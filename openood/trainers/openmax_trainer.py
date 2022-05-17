from cProfile import label

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.utils import Config

from .lr_scheduler import cosine_annealing


def adjust_learning_rate(optimizer, epoch, lr, factor=0.1, step=30):
    """Sets the learning rate to the initial LR decayed by factor every step
    epochs."""
    lr = lr * (factor**(epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class OpenMaxTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.lr = config.optimizer.lr

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, epoch_idx):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0

        adjust_learning_rate(optimizer=self.optimizer,
                             epoch=epoch_idx,
                             lr=self.lr)
        train_dataiter = iter(self.train_loader)
        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            self.optimizer.zero_grad()
            outputs = self.net(data)

            loss = self.criterion(outputs, target)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        loss_avg = train_loss / len(train_dataiter)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_avg
        metrics['acc'] = correct / total
        print('training acc: ' + str(metrics['acc']))
        print('\nLearning rate: %f' % (self.optimizer.param_groups[0]['lr']))

        return self.net, metrics
