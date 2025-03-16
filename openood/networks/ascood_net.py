import torch
import torch.nn as nn


def z_std(x):
    mu = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    z_feature = (x - mu) / (std + 1e-6)
    return z_feature


class ASCOODNet(nn.Module):
    def __init__(self, backbone):
        super(ASCOODNet, self).__init__()
        self.backbone = backbone
        self.register_buffer('sigma', torch.tensor(1.0))
        self.feature_size = backbone.feature_size

    def set_params(self, sigma):
        self.sigma.fill_(sigma)

    def forward(self, x, return_feature=False):
        _, feature = self.backbone(x, return_feature=True)
        pre_feature = feature.clone()
        feature = z_std(feature) * self.sigma
        output = self.get_fc_layer()(feature)
        if return_feature:
            return output, pre_feature
        return output

    def get_fc_layer(self):
        return self.backbone.fc
