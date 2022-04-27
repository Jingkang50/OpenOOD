import torch
import torch.backends.cudnn as cudnn

from .densenet import DenseNet3
from .draem_networks import DiscriminativeSubNetwork, ReconstructiveSubNetwork
from .lenet import LeNet
from .opengan import Discriminator, Generator
from .projectionnet import ProjectionNet
from .resnet18_32x32 import ResNet18_32x32
from .resnet18_224x224 import ResNet18_224x224
from .vggnet import Vgg16, make_arch
from .wrn import WideResNet


def get_network(network_config):

    num_classes = network_config.num_classes

    if network_config.name == 'resnet18_32x32':
        net = ResNet18_32x32(num_classes=num_classes)

    elif network_config.name == 'resnet18_224x224':
        net = ResNet18_224x224(num_classes=num_classes)

    elif network_config.name == 'lenet':
        net = LeNet(num_classes=num_classes, num_channel=3)

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

        net = {'generative': model, 'discriminative': model_seg}

    elif network_config.name == 'opengan':
        # NetType = eval(network_config.feat_extract_network)
        # feature_net = NetType()
        feature_net = get_network(network_config.feat_extract_network)

        netG = Generator(in_channels=network_config.nz,
                         feature_size=network_config.ngf,
                         out_channels=network_config.nc)
        netD = Discriminator(in_channels=network_config.nc,
                             feature_size=network_config.ndf)

        net = {'netG': netG, 'netD': netD, 'netF': feature_net}

    elif network_config.name == 'vgg and model':
        vgg = Vgg16(network_config['trainedsource'])
        model = make_arch(network_config['equal_network_size'],
                          network_config['use_bias'], True)
        net = {'vgg': vgg, 'model': model}

    elif network_config.name == 'cutpaste':
        net = ProjectionNet(num_classes=2)

    else:
        raise Exception('Unexpected Network Architecture!')

    if network_config.pretrained:
        if type(net) is dict:
            for subnet, checkpoint in zip(net.values(),
                                          network_config.checkpoint):
                if checkpoint is not None:
                    if checkpoint != 'none':
                        subnet.load_state_dict(torch.load(checkpoint),
                                               strict=False)
        else:
            try:
                net.load_state_dict(torch.load(network_config.checkpoint),
                                    strict=False)
            except RuntimeError:
                # sometimes fc should not be loaded
                loaded_pth = torch.load(network_config.checkpoint)
                loaded_pth.pop('fc.weight')
                loaded_pth.pop('fc.bias')
                net.load_state_dict(loaded_pth, strict=False)
        print('Model Loading {} Completed!'.format(network_config.name))

    if network_config.num_gpus > 1:
        if type(net) is dict:
            for key, subnet in zip(net.keys(), net.values()):
                net[key] = torch.nn.DataParallel(
                    subnet, device_ids=list(range(network_config.num_gpus)))
        else:
            net = torch.nn.DataParallel(net,
                                        device_ids=list(
                                            range(network_config.num_gpus)))

    if network_config.num_gpus > 0:
        if type(net) is dict:
            for subnet in net.values():
                subnet.cuda()
        else:
            net.cuda()
        torch.cuda.manual_seed(1)

    cudnn.benchmark = True
    return net
