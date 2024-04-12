import torch.nn as nn

class TimmAdapter(nn.Module):
    def __init__(self, timm_module: nn.Module):
        self.timm_module = timm_module

    def forward(self, x, return_feature=False, return_feature_list=False):
        if return_feature:
            feature = self.timm_module.forward_features(x)
            logits_cls = self.timm_module.forward_head(feature)
            return logits_cls, feature.view(feature.size(0), -1)
        if return_feature_list:
            raise NotImplementedError('return_feature_list is not implemented.')
        return self.timm_module.forward(x)

    # def forward_threshold(self, x, threshold):
    #     feature1 = F.relu(self.bn1(self.conv1(x)))
    #     feature2 = self.layer1(feature1)
    #     feature3 = self.layer2(feature2)
    #     feature4 = self.layer3(feature3)
    #     feature5 = self.layer4(feature4)
    #     feature5 = self.avgpool(feature5)
    #     feature = feature5.clip(max=threshold)
    #     feature = feature.view(feature.size(0), -1)
    #     logits_cls = self.fc(feature)

    #     return logits_cls

    # def intermediate_forward(self, x, layer_index):
    #     return self.forward_features(x)

    # def get_fc(self):
    #     fc = self.fc
    #     return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    # def get_fc_layer(self):
    #     return self.fc

