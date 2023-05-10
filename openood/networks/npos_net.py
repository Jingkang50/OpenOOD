import torch
import torch.nn as nn
import torch.nn.functional as F


class NPOSNet(nn.Module):
    def __init__(self, backbone, head, feat_dim, num_classes):
        super(NPOSNet, self).__init__()

        self.backbone = backbone
        if hasattr(self.backbone, 'fc'):
            # remove fc otherwise ddp will
            # report unused params
            self.backbone.fc = nn.Identity()

        try:
            feature_size = backbone.feature_size
        except AttributeError:
            feature_size = backbone.module.feature_size

        self.prototypes = nn.Parameter(torch.zeros(num_classes, feat_dim),
                                       requires_grad=True)
        self.mlp = nn.Sequential(nn.Linear(feature_size, feat_dim),
                                 nn.ReLU(inplace=True), nn.Linear(feat_dim, 1))

        if head == 'linear':
            self.head = nn.Linear(feature_size, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(nn.Linear(feature_size, feature_size),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(feature_size, feat_dim))

    def forward(self, x):
        feat = self.backbone(x).squeeze()
        unnorm_features = self.head(feat)
        features = F.normalize(unnorm_features, dim=1)
        return features

    def intermediate_forward(self, x):
        feat = self.backbone(x).squeeze()
        return F.normalize(feat, dim=1)
