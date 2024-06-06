import torch
import torch.nn as nn
import torch.nn.functional as F


class T2FNormNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(T2FNormNet, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.register_buffer('tau', torch.tensor(1.0))

    def set_tau(self, tau):
        self.tau = torch.tensor(tau)

    def forward(self, x):
        _, feature = self.backbone(x, return_feature=True)
        feature = F.normalize(feature, dim=-1) / self.tau
        output = self.backbone.fc(feature)
        return output

    def forward_ood_inference(self, x):
        _, feature = self.backbone(x, return_feature=True)
        if self.num_classes != 1000:
            # Imagenet-1k experiment is not trained from scratch.
            # Use temperature scaling only for models trained from scratch.
            feature = feature / self.tau
        output = self.backbone.fc(feature)
        return output
