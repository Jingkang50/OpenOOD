import torch
import torch.backends.cudnn as cudnn
from torch import nn

from .densenet import DenseNet3
from .draem_networks import DiscriminativeSubNetwork, ReconstructiveSubNetwork
from .lenet import LeNet
from .resnet18 import ResNet18
from .resnet18L import ResNet18L
from .vggnet import Vgg16, load_network, make_arch
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

    elif network_config.name == 'DRAEM':
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        if network_config.pretrained:
            model.load_state_dict(
                torch.load(network_config.checkpoint + '.ckpt',
                           map_location='cuda:0'))
            model_seg.load_state_dict(
                torch.load(network_config.checkpoint + '_seg.ckpt',
                           map_location='cuda:0'))
        if network_config.num_gpus > 1:
            pass
        if network_config.num_gpus > 0:
            model.cuda()
            model_seg.cuda()
            torch.cuda.manual_seed(1)
        return {'generative': model, 'discriminative': model_seg}
    elif network_config.name == 'vgg and model':
        if network_config['equal_network_size']:
            config_type = 'A'
        else:
            config_type = 'B'

        vgg = Vgg16(network_config['pretrained']).cuda()
        model = make_arch(config_type, network_config['use_bias'], True).cuda()

        for j, item in enumerate(nn.ModuleList(model.features)):
            print('layer : {} {}'.format(j, item))

        load_network(network_config, vgg, model)
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
