from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class DRAEMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(DRAEMPostprocessor, self).__init__(config)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # forward
        gray_rec = net['generative'](data)
        joined_in = torch.cat((gray_rec.detach(), data), dim=1)

        out_mask = net['discriminative'](joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)

        # calculate image level scores
        out_mask_averaged = torch.nn.functional.avg_pool2d(
            out_mask_sm[:, 1:, :, :], 21, stride=1,
            padding=21 // 2).cpu().detach().numpy()

        image_score = np.max(out_mask_averaged, axis=(1, 2, 3))

        return -1 * torch.ones(data.shape[0]), torch.tensor(
            [-image_score]).reshape((data.shape[0]))
