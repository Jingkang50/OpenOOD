from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F

from .base_postprocessor import BasePostprocessor


class Rd4adPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(Rd4adPostprocessor, self).__init__(config)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        encoder = net['encoder']
        bn = net['bn']
        decoder = net['decoder']
        feature_list = encoder.forward(data, return_feature_list=True)[1]
        input = feature_list[1:4]
        output = decoder(bn(input))
        anomaly_map, _ = cal_anomaly_map(input,
                                         output,
                                         data.shape[-1],
                                         amap_mode='a')
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)

        conf = np.max(anomaly_map)

        return -1 * torch.ones(data.shape[0]), torch.tensor([-conf]).reshape(
            (data.shape[0]))

    # def inference(self, net: nn.Module, data_loader: DataLoader):
    #     pred_list, conf_list, label_list = [], [], []
    #     for batch in data_loader:
    #         data = batch['data'].cuda()
    #         label = batch['label'].cuda()
    #         import pdb
    #         pdb.set_trace()
    #         conf = self.postprocess(net, data)
    #         for idx in range(len(data)):
    #             conf_list.append(conf[idx].tolist())
    #             label_list.append(label[idx].cpu().tolist())

    #     # convert values into numpy array

    #     conf_list = np.array(conf_list)
    #     label_list = np.array(label_list, dtype=int)

    #     return pred_list, conf_list, label_list


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]

        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map,
                              size=out_size,
                              mode='bilinear',
                              align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list
