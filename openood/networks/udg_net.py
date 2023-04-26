import torch.nn as nn


class UDGNet(nn.Module):
    def __init__(self, backbone, num_classes, num_clusters):
        super(UDGNet, self).__init__()
        self.backbone = backbone
        if hasattr(self.backbone, 'fc'):
            # remove fc otherwise ddp will
            # report unused params
            self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(backbone.feature_size, num_classes)
        self.fc_aux = nn.Linear(backbone.feature_size, num_clusters)

    def forward(self, x, return_feature=False, return_aux=False):
        _, feature = self.backbone(x, return_feature=True)
        logits_cls = self.fc(feature)
        logits_aux = self.fc_aux(feature)

        if return_aux:
            if return_feature:
                return logits_cls, logits_aux, feature
            else:
                return logits_cls, logits_aux
        else:
            if return_feature:
                return logits_cls, feature
            else:
                return logits_cls
