import torch.nn as nn


class conf_net(nn.Module):
    def __init__(self, backbone, num_classes, num_channel=3):
        super(conf_net, self).__init__()

        self.backbone = backbone
        self.fc = nn.Linear(10, num_classes)
        self.confidence = nn.Linear(10, 1)

    # test conf
    def forward(self, x, return_confidence=False):

        logits_cls = self.backbone(x,
                                   return_feature=False,
                                   return_feature_list=False)

        pred = self.fc(logits_cls)
        confidence = self.confidence(logits_cls)

        if return_confidence:
            return pred, confidence
        else:
            return pred
