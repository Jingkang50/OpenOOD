import torch.nn as nn
from torchvision.models import resnet18


class ProjectionNet(nn.Module):
    def __init__(self,
                 backbone,
                 head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128],
                 num_classes=2):
        super(ProjectionNet, self).__init__()
        self.backbone = backbone

        # use res18 pretrained model if none is given
        # self.backbone=resnet18(pretrained=True)

        # penultimate layer feature size
        last_layer = backbone.feature_size
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons

        # the last layer without activation
        head = nn.Sequential(*sequential_layers)
        self.head = head
        self.out = nn.Linear(last_layer, num_classes)

    def forward(self, x):
        # penultimate layer feature
        _, embeds = self.backbone(x, return_feature=True)
        tmp = self.head(embeds)
        logits = self.out(tmp)
        return embeds, logits

    def get_fc(self):
        fc = self.out
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()
