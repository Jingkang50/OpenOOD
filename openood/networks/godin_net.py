import torch
import torch.nn as nn


def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x


class CosineDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CosineDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias=False)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity='relu')

    def forward(self, x):
        x = norm(x)
        w = norm(self.h.weight)

        ret = (torch.matmul(x, w.T))
        return ret


class EuclideanDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(EuclideanDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias=False)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity='relu')

    def forward(self, x):

        # size: (batch, latent, 1)
        x = x.unsqueeze(2)

        # size: (1, latent, num_classes)
        h = self.h.weight.T.unsqueeze(0)
        ret = -((x - h).pow(2)).mean(1)
        return ret


class InnerDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(InnerDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity='relu')
        self.h.bias.data = torch.zeros(size=self.h.bias.size())

    def forward(self, x):
        return self.h(x)


class GodinNet(nn.Module):
    def __init__(self,
                 backbone,
                 feature_size,
                 num_classes,
                 similarity_measure='cosine'):
        super(GodinNet, self).__init__()

        h_dict = {
            'cosine': CosineDeconf,
            'inner': InnerDeconf,
            'euclid': EuclideanDeconf
        }

        self.num_classes = num_classes

        self.backbone = backbone
        if hasattr(self.backbone, 'fc'):
            # remove fc otherwise ddp will
            # report unused params
            self.backbone.fc = nn.Identity()

        self.h = h_dict[similarity_measure](feature_size, num_classes)

        self.g = nn.Sequential(nn.Linear(feature_size, 1), nn.BatchNorm1d(1),
                               nn.Sigmoid())

        self.softmax = nn.Softmax()

    def forward(self, x, inference=False, score_func='h'):
        _, feature = self.backbone(x, return_feature=True)

        numerators = self.h(feature)

        denominators = self.g(feature)

        # calculate the logits results
        quotients = numerators / denominators

        # logits, numerators, and denominators
        if inference:
            if score_func == 'h':
                return numerators
            elif score_func == 'g':
                return denominators
            else:
                # maybe generate an error instead
                print('Invalid score function, using h instead')
                return numerators
        else:
            return quotients
