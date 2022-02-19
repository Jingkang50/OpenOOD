from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch import nn

from .densenet import DenseNet3
from .lenet import LeNet
from .resnet18 import ResNet18
from .resnet18L import ResNet18L
from .vggnet import Vgg16, make_arch
from .wrn import WideResNet


def get_network(network_config):

    num_classes = network_config.num_classes

    if network_config.name == 'res18':
        net = ResNet18(num_classes=num_classes)

    elif network_config.name == 'res18L':
        net = ResNet18L(num_classes=num_classes)

    elif network_config.name == 'lenet':
        net = LeNet(num_classes=num_classes, num_channel=3)

    elif network_config.name == 'lenet_bw':
        net = LeNet(num_classes=num_classes, num_channel=1)

    elif network_config.name == 'wrn':
        net = WideResNet(depth=28,
                         widen_factor=10,
                         dropRate=0.0,
                         num_classes=num_classes)

    elif network_config.name == 'densenet':
        net = DenseNet3(depth=100,
                        growth_rate=12,
                        reduction=0.5,
                        bottleneck=True,
                        dropRate=0.0,
                        num_classes=num_classes)
    elif network_config.name == 'vgg and model':
        equal_network_size = network_config['equal_network_size']
        pretrained = network_config['pretrained']
        experiment_name = network_config['exp_name']
        normal_class = network_config['normal_class']
        use_bias = network_config['use_bias']
        load_checkpoint = network_config['load_checkpoint']
        cfg = {
            'A': [
                64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                'M', 512, 512, 512, 'M'
            ],
            'B': [
                16, 16, 'M', 16, 128, 'M', 16, 16, 256, 'M', 16, 16, 512, 'M',
                16, 16, 512, 'M'
            ],
        }
        if equal_network_size:
            config_type = 'A'

        else:
            config_type = 'B'

        vgg = Vgg16(pretrained).cuda()
        model = make_arch(config_type, cfg, use_bias, True).cuda()

        for j, item in enumerate(nn.ModuleList(model.features)):
            print('layer : {} {}'.format(j, item))

        if load_checkpoint:
            last_checkpoint = network_config['last_checkpoint']
            checkpoint_path = './results/{}/'.format(experiment_name)

            model.load_state_dict(
                torch.load('{}Cloner_{}_epoch_{}.pth'.format(
                    checkpoint_path, normal_class, last_checkpoint)))
            if not pretrained:
                vgg.load_state_dict(
                    torch.load('{}Source_{}_random_vgg.pth'.format(
                        checkpoint_path, normal_class)))
        elif not pretrained:
            checkpoint_path = './results/{}/'.format(experiment_name)

            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

            torch.save(
                vgg.state_dict(),
                '{}Source_{}_random_vgg.pth'.format(checkpoint_path,
                                                    normal_class))
            print('Source Checkpoint saved!')
        net = {}
        net['vgg'] = vgg
        net['model'] = model

        return net

    else:
        raise Exception('Unexpected Network Architecture!')

    if network_config.pretrained:
        net.load_state_dict(torch.load(network_config.checkpoint),
                            strict=False)
        print('Model Loading Completed!')

    if network_config.num_gpus > 1:
        net = torch.nn.DataParallel(net,
                                    device_ids=list(
                                        range(network_config.num_gpus)))

    if network_config.num_gpus > 0:
        net.cuda()
        torch.cuda.manual_seed(1)

    cudnn.benchmark = True
    return net
