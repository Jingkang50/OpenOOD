import os

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.postprocessors import BasePostprocessor
from openood.utils import Config
from .gaussian_density import GaussianDensityTorch
from openood.postprocessors.cp_tools import get_train_embeds


def to_np(x):
    return x.data.cuda().numpy()


class CutPasteEvaluator:
    def __init__(self, config: Config):
        self.config = config

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        net.eval()

        # loss_avg = 0.0
        correct = 0
        
        embeds = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True):
                # prepare data
                data = torch.cat(batch['data'], 0)
                data = data.cuda()
                # target = batch['label'].cuda()
                
                # calculate label
                y = torch.arange(2)
                y = y.repeat_interleave(len(batch['data'][0]))
                
                # forward
                embed, output = net(data)
                # loss = F.cross_entropy(output, target)
                embeds.append(embed.cuda())

                y = y.cuda()
                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(y.data).sum().item()

                # test loss average
                # loss_avg += float(distances.data)

        embeds = torch.cat(embeds)
        embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
        
        train_embeds = get_train_embeds(net, self.config)
        train_embeds = torch.nn.functional.normalize(train_embeds, p=2, dim=1)
        density = GaussianDensityTorch()
        density.fit(train_embeds)
        distances = density.predict(embeds)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = distances
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
