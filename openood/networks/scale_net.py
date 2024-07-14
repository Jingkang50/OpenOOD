import numpy as np
import torch
import torch.nn as nn


class ScaleNet(nn.Module):
    def __init__(self, backbone):
        super(ScaleNet, self).__init__()
        self.backbone = backbone

    def forward(self, x, return_feature=False, return_feature_list=False):
        try:
            return self.backbone(x, return_feature, return_feature_list)
        except TypeError:
            return self.backbone(x, return_feature)

    def forward_threshold(self, x, percentile):
        _, feature = self.backbone(x, return_feature=True)
        feature = scale(feature.view(feature.size(0), -1, 1, 1), percentile)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.backbone.get_fc_layer()(feature)
        return logits_cls

    def get_fc(self):
        fc = self.backbone.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()


def scale(x, percentile=65):
    input = x.clone()
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2

    return input * torch.exp(scale[:, None, None, None])
