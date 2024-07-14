import torch.nn as nn
import torch.nn.functional as F


class SimClrNet(nn.Module):
    def __init__(self, backbone, out_dim=128) -> None:
        super(SimClrNet, self).__init__()

        self.backbone = backbone
        feature_dim = backbone.feature_size
        self.simclr_head = nn.Sequential(nn.Linear(feature_dim, feature_dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(feature_dim, out_dim))

    def forward(self, x, return_feature=False, return_feature_list=False):
        _, feature = self.backbone.forward(x, return_feature=True)

        return _, [F.normalize(self.simclr_head(feature), dim=-1)]
