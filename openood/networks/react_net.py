import torch.nn as nn


class ReactNet(nn.Module):
    def __init__(self, backbone):
        super(ReactNet, self).__init__()
        self.backbone = backbone

    def forward(self, x, return_feature=False, return_feature_list=False):
        return self.backbone(x, return_feature, return_feature_list)

    def forward_threshold(self, x, threshold):
        _, feature_list = self.backbone(x, return_feature_list=True)
        feature = feature_list[-1]
        feature = feature.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.backbone.fc(feature)
        return logits_cls

    def get_fc(self):
        fc = self.backbone.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()
