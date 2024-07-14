import torch

from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
from torch.nn import functional as F
from tqdm import tqdm
from openood.utils import Config
from openood.losses.rd4ad_loss import loss_function


class Rd4adTrainer:
    def __init__(self, net, train_loader, config: Config):
        self.config = config
        self.train_loader = train_loader
        self.encoder = net['encoder']
        self.bn = net['bn']
        self.decoder = net['decoder']
        if config.optimizer.name == 'adam':
            self.optimizer = torch.optim.Adam(list(self.decoder.parameters()) +
                                              list(self.bn.parameters()),
                                              lr=config.optimizer.lr,
                                              betas=config.optimizer.betas)

    def train_epoch(self, epoch_idx):
        self.encoder.eval()
        self.bn.train()
        self.decoder.train()
        train_dataiter = iter(self.train_loader)
        epoch_loss = 0
        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d} '.format(epoch_idx),
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            img = batch['data'].cuda()
            feature_list = self.encoder.forward(img,
                                                return_feature_list=True)[1]
            inputs = feature_list[1:4]
            outputs = self.decoder(self.bn(inputs))
            loss = loss_function(inputs, outputs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        metrics = {}
        net = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = epoch_loss
        net['encoder'] = self.encoder
        net['bn'] = self.bn
        net['decoder'] = self.decoder
        return net, metrics
