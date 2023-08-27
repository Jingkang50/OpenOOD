import torch
import torch.nn as nn
import torch.nn.functional as F

class T2FNormNet(nn.Module):
    def __init__(self, backbone, tau, num_classes):
        super(T2FNormNet, self).__init__()

        self.register_buffer('tau', torch.tensor(tau))
        self.backbone = backbone
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()

        try:
            feature_size = backbone.feature_size
        except AttributeError:
            feature_size = backbone.module.feature_size

        self.new_fc = nn.Linear(feature_size, num_classes)

    def forward(self, x, return_feature=False):
        penultimate_features = self.backbone(x)
        if self.training:
            features = F.normalize(penultimate_features, dim=-1) / self.tau.item()
        else:
            features = penultimate_features / self.tau.item()

        logits_cls = self.new_fc(features)
        if return_feature:
            return logits_cls, penultimate_features
        else:
            return logits_cls

    def intermediate_forward(self, x):
        penultimate_features = self.backbone(x).squeeze()
        return penultimate_features
