import torch.nn as nn


class RotNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(RotNet, self).__init__()

        self.backbone = backbone

        self.fc = nn.Linear(backbone.feature_size, num_classes)
        self.rot_fc = nn.Linear(backbone.feature_size, 4)

    def forward(self, x, return_rot_logits=False):
        _, feature = self.backbone(x, return_feature=True)

        logits = self.fc(feature)
        rot_logits = self.rot_fc(feature)

        if return_rot_logits:
            return logits, rot_logits
        else:
            return logits
