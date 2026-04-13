import torch
import torch.nn as nn


class RASNet(nn.Module):
    def __init__(self, backbone):
        super(RASNet, self).__init__()
        self.backbone = backbone
        self.mean_curve = None

    def forward(self, x, return_feature=False, return_feature_list=False):
        try:
            return self.backbone(x, return_feature, return_feature_list)
        except TypeError:
            return self.backbone(x, return_feature)

    def forward_shift(self, x):
        _, feature = self.backbone(x, return_feature=True)
        feature = feature.view(feature.size(0), -1)

        sorted_vals, idx = torch.sort(feature, dim=1)
        mc = self.mean_curve.to(feature.device).expand_as(sorted_vals)
        shifted = torch.empty_like(feature).scatter_(1, idx, mc)

        logits_cls = self.backbone.get_fc_layer()(shifted)
        return logits_cls

    def set_mean_curve(self, curve):
        self.mean_curve = curve

    def get_fc(self):
        fc = self.backbone.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()
