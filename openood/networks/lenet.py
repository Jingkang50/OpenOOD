import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


class LeNet(nn.Module):
    def __init__(self, num_classes, num_channel=3):
        super(LeNet, self).__init__()
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channel,
                      out_channels=6,
                      kernel_size=5,
                      stride=1,
                      padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=120,
                      kernel_size=5,
                      stride=1), nn.ReLU())

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=num_classes),
        )
        self.feature_size = 120

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature1 = self.block1(x)
        feature2 = self.block2(feature1)
        feature3 = self.block3(feature2)
        feature = feature3.view(feature3.shape[0], -1)
        logits_cls = self.classifier(feature)
        feature_list = [feature1, feature2, feature3]
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls
