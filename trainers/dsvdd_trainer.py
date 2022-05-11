import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from openood.utils import Config


class AETrainer:
    def __init__(self, net, train_loader, config: Config):
        self.net = net.to(config.device)
        self.config = config
        self.train_loader = train_loader
        if config.optimizer.name == 'adam':
            self.optimizer = optim.Adam(
                net.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                amsgrad=config.optimizer.name == 'amsgrad')
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config.lr_milestones, gamma=0.1)

    def train_epoch(self, epoch_idx):

        self.net.train()
        epoch_loss = 0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d} '.format(epoch_idx),
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            inputs = batch['data'].to(self.config.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            scores = torch.sum((outputs - inputs)**2,
                               dim=tuple(range(1, outputs.dim())))
            loss = torch.mean(scores)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            epoch_loss += loss.item()
        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['epoch_loss'] = epoch_loss
        return self.net, metrics


class DSVDDTrainer:
    def __init__(self, net, train_loader, config: Config) -> None:
        self.net = net.to(config.device)
        self.config = config
        self.train_loader = train_loader
        if config.optimizer.name == 'adam':
            self.optimizer = optim.Adam(
                net.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                amsgrad=config.optimizer.name == 'amsgrad')
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config.lr_milestones, gamma=0.1)

    def train_epoch(self, epoch_idx):
        R = self.config.R
        if self.config.c == 'None':
            c = init_center_c(self.train_loader, self.net)

        self.net.train()
        epoch_loss = 0
        train_dataiter = iter(self.train_loader)
        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}'.format(epoch_idx),
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            inputs = batch['data'].to(self.config.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            dist = torch.sum((outputs - c)**2, dim=1)
            if self.config.objective == 'soft-boundary':
                scores = dist - R**2
                loss = R**2 + (1 / self.nu) * torch.mean(
                    torch.max(torch.zeros_like(scores), scores))
            else:
                loss = torch.mean(dist)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if (self.config.objective == 'soft-boundary') and (
                    epoch_idx >= self.config.warm_up_n_epochs):
                R.data = torch.tensor(get_radius(dist, self.config.nu),
                                      device=self.config.device)
            epoch_loss += loss.item()
        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['epoch_loss'] = epoch_loss
        hyperparas = {}
        hyperparas['R'] = R
        hyperparas['c'] = c
        return self.net, metrics, hyperparas


def init_center_c(train_loader, net, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass
    on the data."""
    n_samples = 0
    c = torch.zeros(net.rep_dim, device=torch.device('cuda'))
    train_dataiter = iter(train_loader)
    net.eval()
    with torch.no_grad():
        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Initialize center',
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            inputs = batch['data'].cuda()
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps.
    # Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
