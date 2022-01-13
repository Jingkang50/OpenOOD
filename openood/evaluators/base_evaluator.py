import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.utils import Config


def to_np(x):
    return x.data.cpu().numpy()


class BaseEvaluator:
    def __init__(self, config: Config):
        self.config = config

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 epoch_idx: int = -1):
        net.eval()

        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                target = batch['label'].cuda()

                # forward
                output = net(data)
                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_avg / len(data_loader)
        metrics['acc'] = correct / len(data_loader.dataset)
        return metrics

    def extract(self, net: nn.Module, data_loader: DataLoader):
        net.eval()
        feat_list, label_list = [], []

        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Feature Extracting: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                label = batch['label']

                _, feat = net(data, return_feature=True)
                feat_list.extend(to_np(feat))
                label_list.extend(to_np(label))

        feat_list = np.array(feat_list)
        label_list = np.array(label_list)

        save_dir = self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, 'feature'),
                 feat_list=feat_list,
                 label_list=label_list)
