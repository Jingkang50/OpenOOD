import torch.nn as nn
import torch.nn.functional as F


class OpenMax(nn.Module):
    def __init__(self, backbone, num_classes=50, embed_dim=None):
        super(OpenMax, self).__init__()
        self.backbone_name = backbone
        self.backbone = backbone

        self.dim = self.get_backbone_last_layer_out_channel()
        if embed_dim:
            self.embeddingLayer = nn.Sequential(
                nn.Linear(self.dim, embed_dim),
                nn.PReLU(),
            )
            self.dim = embed_dim
        self.classifier = nn.Linear(self.dim, num_classes)

    def get_backbone_last_layer_out_channel(self):
        if self.backbone_name == 'LeNetPlus':
            return 128 * 3 * 3
        last_layer = list(self.backbone.children())[-1]
        while (not isinstance(last_layer, nn.Conv2d)) and \
                (not isinstance(last_layer, nn.Linear)) and \
                (not isinstance(last_layer, nn.BatchNorm2d)):

            temp_layer = list(last_layer.children())[-1]
            if isinstance(temp_layer, nn.Sequential) and len(
                    list(temp_layer.children())) == 0:
                temp_layer = list(last_layer.children())[-2]
            last_layer = temp_layer
        if isinstance(last_layer, nn.BatchNorm2d):
            return last_layer.num_features
        elif isinstance(last_layer, nn.Linear):
            return last_layer.out_features
        else:
            return last_layer.out_channels

    def forward(self, x):
        feature = self.backbone(x)
        if feature.dim() == 4:
            feature = F.adaptive_avg_pool2d(feature, 1)
            feature = feature.view(x.size(0), -1)
        # if includes embedding layer.
        feature = self.embeddingLayer(feature) if hasattr(
            self, 'embeddingLayer') else feature
        logits = self.classifier(feature)

        return logits

    def get_fc(self):
        fc = self.classifier
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()
