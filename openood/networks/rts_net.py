import torch
import torch.nn as nn


class RTSNet(nn.Module):
    def __init__(self, backbone, feature_size, num_classes, dof=16):
        '''
        dof: degree of freedom of variance
        '''
        super(RTSNet, self).__init__()
        self.backbone = backbone
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.dof = dof
        self.logvar_rts = nn.Sequential(
            nn.Linear(feature_size, self.dof),
            nn.BatchNorm1d(self.dof),
        )

    def forward(self, x, return_var=False):
        logits_cls, feature = self.backbone(x, return_feature=True)
        if return_var:
            logvar = self.logvar_rts(feature)
            variance = logvar.exp()
            return logits_cls, variance
        else:
            return logits_cls
