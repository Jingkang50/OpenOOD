from pathlib import Path

import torch
from torch import nn
from torchvision.models import vgg16


class VGG(nn.Module):
    """VGG model."""
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

        # placeholder for the gradients
        self.gradients = None
        self.activation = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, target_layer=11):
        result = []
        for i in range(len(nn.ModuleList(self.features))):
            x = self.features[i](x)
            if i == target_layer:
                self.activation = x
            if i in [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38]:
                result.append(x)
        return result

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.activation


def make_layers(cfg, use_bias, batch_norm=False):
    layers = []
    in_channels = 3
    outputs = []
    for i in range(len(cfg)):
        if cfg[i] == 'O':
            outputs.append(nn.Sequential(*layers))
        elif cfg[i] == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels,
                               cfg[i],
                               kernel_size=3,
                               padding=1,
                               bias=use_bias)
            torch.nn.init.xavier_uniform_(conv2d.weight)
            if batch_norm and cfg[i + 1] != 'M':
                layers += [
                    conv2d,
                    nn.BatchNorm2d(cfg[i]),
                    nn.ReLU(inplace=True)
                ]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = cfg[i]
    return nn.Sequential(*layers)


def make_arch(equal_size, use_bias, batch_norm=False):
    if equal_size:
        idx = 'A'
    else:
        idx = 'B'
    """for clone network."""
    cfg = {
        'A': [
            64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
            512, 512, 512, 'M'
        ],
        'B': [
            16, 16, 'M', 16, 128, 'M', 16, 16, 256, 'M', 16, 16, 512, 'M', 16,
            16, 512, 'M'
        ],
    }
    return VGG(make_layers(cfg[idx], use_bias, batch_norm=batch_norm))


class Vgg16(torch.nn.Module):
    def __init__(self, pretrain):
        super(Vgg16, self).__init__()
        features = list(vgg16('vgg16-397923af.pth').features)

        if not pretrain:
            for ind, f in enumerate(features):
                # nn.init.xavier_normal_(f)
                if type(f) is torch.nn.modules.conv.Conv2d:
                    torch.nn.init.xavier_uniform(f.weight)
                    print('Initialized', ind, f)
                else:
                    print('Bypassed', ind, f)
            # print("Pre-trained Network loaded")
        self.features = nn.ModuleList(features).eval()
        self.output = []

    def forward(self, x):
        output = []
        for i in range(31):
            x = self.features[i](x)
            if i in [1, 4, 6, 9, 11, 13, 16, 18, 20, 23, 25, 27, 30]:
                output.append(x)
        return output


def load_network(network_config, vgg, model):

    if network_config['load_checkpoint']:
        last_checkpoint = network_config['last_checkpoint']
        checkpoint_path = './results/{}/'.format(network_config['exp_name'])

        model.load_state_dict(
            torch.load('{}Cloner_{}_epoch_{}.pth'.format(
                checkpoint_path, network_config['normal_class'],
                last_checkpoint)))
        if not network_config['pretrained']:
            vgg.load_state_dict(
                torch.load('{}Source_{}_random_vgg.pth'.format(
                    checkpoint_path, network_config['normal_class'])))
    elif not network_config['pretrained']:
        checkpoint_path = './results/{}/'.format(network_config['exp_name'])

        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        torch.save(
            vgg.state_dict(), '{}Source_{}_random_vgg.pth'.format(
                checkpoint_path, network_config['normal_class']))
        print('Source Checkpoint saved!')
