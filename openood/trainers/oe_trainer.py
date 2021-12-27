import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from openood.losses import soft_cross_entropy

from .base_trainer import BaseTrainer


class OETrainer(BaseTrainer):
    def __init__(
        self,
        net: nn.Module,
        labeled_train_loader: DataLoader,
        unlabeled_train_loader: DataLoader,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        epochs: int = 100,
        lambda_oe: float = 0.5,
    ) -> None:
        super().__init__(
            net,
            labeled_train_loader,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            epochs=epochs,
        )

        self.unlabeled_train_loader = unlabeled_train_loader
        self.lambda_oe = lambda_oe

    def train_epoch(self):
        self.net.train()  # enter train mode

        loss_avg = 0.0
        train_dataiter = iter(self.labeled_train_loader)

        if self.unlabeled_train_loader:
            unlabeled_dataiter = iter(self.unlabeled_train_loader)

        for train_step in range(1, len(train_dataiter) + 1):
            batch = next(train_dataiter)

            data = batch['data'].cuda()
            # forward
            logits_classifier = self.net(data)
            loss = F.cross_entropy(logits_classifier, batch['label'].cuda())

            if self.unlabeled_train_loader:
                try:
                    unlabeled_batch = next(unlabeled_dataiter)
                except StopIteration:
                    unlabeled_dataiter = iter(self.unlabeled_train_loader)
                    unlabeled_batch = next(unlabeled_dataiter)

                unlabeled_data = unlabeled_batch['data'].cuda()

                logits_oe = self.net(unlabeled_data)
                loss_oe = soft_cross_entropy(
                    logits_oe, unlabeled_batch['soft_label'].cuda())

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
        metrics['train_loss'] = loss_avg

        return metrics
