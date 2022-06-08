import torch.nn as nn


class MCDNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(MCDNet, self).__init__()

        self.backbone = backbone

        self.fc1 = nn.Linear(backbone.feature_size, num_classes)
        self.fc2 = nn.Linear(backbone.feature_size, num_classes)

    # test conf
    def forward(self, x, return_double=False):

        _, feature = self.backbone(x, return_feature=True)

        logits1 = self.fc1(feature)
        logits2 = self.fc2(feature)

        if return_double:
            return logits1, logits2
        else:
            return logits1
