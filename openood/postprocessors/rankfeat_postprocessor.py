from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class RankFeatPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(RankFeatPostprocessor, self).__init__(config)
        self.config = config
        self.args = self.config.postprocessor.postprocessor_args

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        inputs = data.cuda()

        # Logit of Block 4 feature
        feat1 = net.intermediate_forward(inputs, layer_index=4)
        B, C, H, W = feat1.size()
        feat1 = feat1.view(B, C, H * W)
        if self.args.accelerate:
            feat1 = feat1 - power_iteration(feat1, iter=20)
        else:
            u, s, v = torch.linalg.svd(feat1, full_matrices=False)
            feat1 = feat1 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(
                v[:, 0:1, :])
        feat1 = feat1.view(B, C, H, W)
        logits1 = net.fc(torch.flatten(net.avgpool(feat1), 1))

        # Logit of Block 3 feature
        feat2 = net.intermediate_forward(inputs, layer_index=3)
        B, C, H, W = feat2.size()
        feat2 = feat2.view(B, C, H * W)
        if self.args.accelerate:
            feat2 = feat2 - power_iteration(feat2, iter=20)
        else:
            u, s, v = torch.linalg.svd(feat2, full_matrices=False)
            feat2 = feat2 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(
                v[:, 0:1, :])
        feat2 = feat2.view(B, C, H, W)
        feat2 = net.layer4(feat2)
        logits2 = net.fc(torch.flatten(net.avgpool(feat2), 1))

        # Fusion at the logit space
        logits = (logits1 + logits2) / 2
        conf = self.args.temperature * torch.logsumexp(
            logits / self.args.temperature, dim=1)

        _, pred = torch.max(logits, dim=1)
        return pred, conf


def _l2normalize(v, eps=1e-10):
    return v / (torch.norm(v, dim=2, keepdim=True) + eps)


# Power Iteration as SVD substitute for acceleration
def power_iteration(A, iter=20):
    u = torch.FloatTensor(1, A.size(1)).normal_(0, 1).view(
        1, 1, A.size(1)).repeat(A.size(0), 1, 1).to(A)
    v = torch.FloatTensor(A.size(2),
                          1).normal_(0, 1).view(1, A.size(2),
                                                1).repeat(A.size(0), 1,
                                                          1).to(A)
    for _ in range(iter):
        v = _l2normalize(u.bmm(A)).transpose(1, 2)
        u = _l2normalize(A.bmm(v).transpose(1, 2))
    sigma = u.bmm(A).bmm(v)
    sub = sigma * u.transpose(1, 2).bmm(v.transpose(1, 2))
    return sub
