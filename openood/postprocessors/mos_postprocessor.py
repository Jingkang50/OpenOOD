from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


def get_group_slices(classes_per_group):
    group_slices = []
    start = 0
    for num_cls in classes_per_group:
        end = start + num_cls + 1
        group_slices.append([start, end])
        start = end
    return torch.LongTensor(group_slices)


def cal_ood_score(logits, group_slices):
    num_groups = group_slices.shape[0]

    all_group_ood_score_MOS = []

    smax = torch.nn.Softmax(dim=-1).cuda()
    for i in range(num_groups):
        group_logit = logits[:, group_slices[i][0]:group_slices[i][1]]

        group_softmax = smax(group_logit)
        group_others_score = group_softmax[:, 0]

        all_group_ood_score_MOS.append(-group_others_score)

    all_group_ood_score_MOS = torch.stack(all_group_ood_score_MOS, dim=1)
    final_max_score_MOS, _ = torch.max(all_group_ood_score_MOS, dim=1)
    return final_max_score_MOS.data.cpu().numpy()


class MOSPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(MOSPostprocessor, self).__init__(config)
        self.config = config
        self.setup_flag = False

    def cal_group_slices(self, train_loader):
        config = self.config
        # if specified group_config
        if (config.trainer.group_config.endswith('npy')):
            classes_per_group = np.load(config.trainer.group_config)
        elif (config.trainer.group_config.endswith('txt')):
            classes_per_group = np.loadtxt(config.trainer.group_config,
                                           dtype=int)
        else:
            # cal group config
            config = self.config
            group = {}
            train_dataiter = iter(train_loader)
            for train_step in tqdm(range(1,
                                         len(train_dataiter) + 1),
                                   desc='cal group_config',
                                   position=0,
                                   leave=True):
                batch = next(train_dataiter)
                group_label = batch['group_label'].cuda()
                class_label = batch['class_label'].cuda()

                for i in range(len(class_label)):
                    try:
                        group[str(
                            group_label[i].cpu().detach().numpy().tolist())]
                    except:
                        group[str(group_label[i].cpu().detach().numpy().tolist(
                        ))] = []

                    if class_label[i].cpu().detach().numpy().tolist() \
                            not in group[str(group_label[i].cpu().detach().numpy().tolist())]:
                        group[str(group_label[i].cpu().detach().numpy().tolist(
                        ))].append(
                            class_label[i].cpu().detach().numpy().tolist())

            classes_per_group = []
            for i in range(len(group)):
                classes_per_group.append(max(group[str(i)]) + 1)

        self.num_groups = len(classes_per_group)
        self.group_slices = get_group_slices(classes_per_group)
        self.group_slices = self.group_slices.cuda()

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # this postprocessor does not really do anything
        # the inference is done in the mos_evaluator
        pass

    def postprocess(self, net: nn.Module, data):
        net.eval()
        confs_mos = []
        with torch.no_grad():

            logits = net(data)
            conf_mos = cal_ood_score(logits, self.group_slices)
            confs_mos.extend(conf_mos)

        # conf = np.array(confs_mos)
        conf = torch.tensor(confs_mos)
        pred = logits.data.max(1)[1]
        return pred, conf
