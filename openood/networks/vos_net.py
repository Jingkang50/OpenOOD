from .lenet import LeNet
import torch.nn as nn

class vos_net(nn.Module):
    def __init__(self, backbone, num_classes, num_channel=3):
        super(vos_net, self).__init__()

        self.backbone = backbone
        self.fc = nn.Linear(120, num_classes)


    # test vos
    def forward(self, x, return_feature=False, return_feature_list=False):
        # feature1 = self.block1(x)
        # feature2 = self.block2(feature1)
        # feature3 = self.block3(feature2)
        # feature = feature3.view(feature3.shape[0], -1)

        # logits_cls = self.classifier(feature)
        # feature_list = [feature1, feature2, feature3]
        
        logits_cls, feature_list = self.backbone(x,return_feature=False,return_feature_list=True)

        return logits_cls, feature_list[len(feature_list)-1].view(-1,120)