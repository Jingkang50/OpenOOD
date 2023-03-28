from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor


def kl_div(d1, d2):
    """Compute KL-Divergence between d1 and d2."""
    dirty_logs = d1 * torch.log2(d1 / d2)
    return torch.sum(torch.where(d1 != 0, dirty_logs, torch.zeros_like(d1)),
                     axis=1)


class RotPredPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(RotPredPostprocessor, self).__init__(config)
        self.config = config

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        batch_size = len(data)

        x_90 = torch.rot90(data, 1, [2, 3])
        x_180 = torch.rot90(data, 2, [2, 3])
        x_270 = torch.rot90(data, 3, [2, 3])

        x_rot = torch.cat([data, x_90, x_180, x_270])
        y_rot = torch.cat([
            torch.zeros(batch_size),
            torch.ones(batch_size),
            2 * torch.ones(batch_size),
            3 * torch.ones(batch_size),
        ]).long().cuda()

        logits, logits_rot = net(x_rot, return_rot_logits=True)
        logits = logits[:batch_size]
        preds = logits.argmax(1)

        # https://github.com/hendrycks/ss-ood/blob/8051356592a152614ab7251fd15084dd86eb9104/multiclass_ood/test_auxiliary_ood.py#L177-L208
        num_classes = logits.shape[1]
        uniform_dist = torch.ones_like(logits) / num_classes
        cls_loss = kl_div(uniform_dist, F.softmax(logits, dim=1))

        rot_one_hot = torch.zeros_like(logits_rot).scatter_(
            1,
            y_rot.unsqueeze(1).cuda(), 1)
        rot_loss = kl_div(rot_one_hot, F.softmax(logits_rot, dim=1))
        rot_0_loss, rot_90_loss, rot_180_loss, rot_270_loss = torch.chunk(
            rot_loss, 4, dim=0)
        total_rot_loss = (rot_0_loss + rot_90_loss + rot_180_loss +
                          rot_270_loss) / 4.0

        # here ID samples will yield larger scores
        scores = cls_loss - total_rot_loss
        return preds, scores
