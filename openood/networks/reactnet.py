import torch.nn as nn


class ReactNet(nn.Module):
    def __init__(self, backbone):
        super(ReactNet, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def forward_threshold(self, x, threshold):
        _, feature_list = self.backbone(x, return_feature_list=True)
        feature = feature_list[-1]
        feature = feature.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.backbone.fc(feature)

        return logits_cls
    
    def get_fc(self):
        fc = self.backbone.fc
        return fc.weight, fc.bias
