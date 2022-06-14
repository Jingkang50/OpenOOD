import torch.nn as nn
import torch.nn.functional as F


class DropoutNet(nn.Module):
    def __init__(self, backbone, dropout_p):
        super(DropoutNet, self).__init__()
        self.backbone = backbone
        self.dropout_p = dropout_p

    def forward(self, x, use_dropout=True):
        if use_dropout:
            return self.forward_with_dropout(x)
        else:
            return self.backbone(x)

    def forward_with_dropout(self, x):
        _, feature = self.backbone(x, return_feature=True)
        feature = F.dropout2d(feature, self.dropout_p, training=True)
        logits_cls = self.backbone.fc(feature)

        return logits_cls
