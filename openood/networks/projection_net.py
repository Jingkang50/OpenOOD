import torch.nn as nn
from torchvision.models import resnet18


class ProjectionNet(nn.Module):
    def __init__(self,
                 pretrained=True,
                 head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128],
                 num_classes=2):
        super(ProjectionNet, self).__init__()
        self.resnet18 = resnet18(pretrained=pretrained)

        last_layer = 512
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons

        # the last layer without activation
        head = nn.Sequential(*sequential_layers)
        self.resnet18.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes)

    def forward(self, x):
        embeds = self.resnet18(x)
        tmp = self.head(embeds)
        logits = self.out(tmp)
        return embeds, logits

    def get_fc(self):
        fc = self.out
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()
