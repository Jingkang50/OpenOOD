from turtle import forward
from types import MethodType

import mmcv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from mmcls.apis import init_model

import openood.utils.comm as comm

from .bit import KNOWN_MODELS
from .conf_widernet import Conf_WideResNet
from .csinet import CsiNet
from .densenet import DenseNet3
from .draem_networks import DiscriminativeSubNetwork, ReconstructiveSubNetwork
from .dsvdd_net import build_network, get_Autoencoder
from .godinnet import GodinNet
from .lenet import LeNet
from .openmax_network import OpenMax
from .patchcore_net import patchcore_net
from .projectionnet import ProjectionNet
from .reactnet import ReactNet
from .resnet18_32x32 import ResNet18_32x32
from .resnet18_224x224 import ResNet18_224x224
from .resnet50 import ResNet50
from .vggnet import Vgg16, make_arch
from .wrn import WideResNet


def get_network(network_config):

    num_classes = network_config.num_classes

    if network_config.name == 'resnet18_32x32':
        net = ResNet18_32x32(num_classes=num_classes)

    elif network_config.name == 'resnet18_224x224':
        net = ResNet18_224x224(num_classes=num_classes)

    elif network_config.name == 'resnet50':
        net = ResNet50(num_classes=num_classes)

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

    elif network_config.name == 'wide_resnet_50_2':
        path = '/home/pengyunwang/.cache/torch/hub/vision-0.9.0'
        module = torch.hub._load_local(path,
                                       'wide_resnet50_2',
                                       pretrained=True)
        net = patchcore_net(module)

    elif network_config.name == 'godinnet':
        backbone = get_network(network_config.backbone)
        net = GodinNet(backbone=backbone,
                       feature_size=backbone.feature_size,
                       num_classes=num_classes,
                       similarity_measure=network_config.similarity_measure)

    elif network_config.name == 'reactnet':
        backbone = get_network(network_config.backbone)
        net = ReactNet(backbone)

    elif network_config.name == 'csinet':
        backbone = get_network(network_config.backbone)
        net = CsiNet(backbone,
                     feature_size=backbone.feature_size,
                     num_classes=num_classes,
                     simclr_dim=network_config.simclr_dim,
                     shift_trans_type=network_config.shift_trans_type)

    elif network_config.name == 'draem':
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)

        net = {'generative': model, 'discriminative': model_seg}

    elif network_config.name == 'openmax_network':
        net = OpenMax(backbone='ResNet18', num_classes=50)

    elif network_config.name == 'opengan':
        from .opengan import Discriminator, Generator
        backbone = get_network(network_config.backbone)
        netG = Generator(in_channels=network_config.nz,
                         feature_size=network_config.ngf,
                         out_channels=network_config.nc)
        netD = Discriminator(in_channels=network_config.nc,
                             feature_size=network_config.ndf)

        net = {'netG': netG, 'netD': netD, 'backbone': backbone}

    elif network_config.name == 'arpl_gan':
        from .arpl_net import (resnet34ABN, Generator, Discriminator,
                               Generator32, Discriminator32)
        from .arpl_layer import ARPLayer
        feature_net = resnet34ABN(num_classes=num_classes, num_bns=2)
        dim_centers = feature_net.fc.weight.shape[1]
        feature_net.fc = nn.Identity()

        criterion = ARPLayer(feat_dim=dim_centers,
                             num_classes=num_classes,
                             weight_pl=network_config.weight_pl,
                             temp=network_config.temp)

        assert network_config.image_size == 32 \
            or network_config.image_size == 64, \
            'ARPL-GAN only supports 32x32 or 64x64 images!'

        if network_config.image_size == 64:
            netG = Generator(1, network_config.nz, network_config.ngf,
                             network_config.nc)  # ngpu, nz, ngf, nc
            netD = Discriminator(1, network_config.nc,
                                 network_config.ndf)  # ngpu, nc, ndf
        else:
            netG = Generator32(1, network_config.nz, network_config.ngf,
                               network_config.nc)  # ngpu, nz, ngf, nc
            netD = Discriminator32(1, network_config.nc,
                                   network_config.ndf)  # ngpu, nc, ndf

        net = {
            'netF': feature_net,
            'criterion': criterion,
            'netG': netG,
            'netD': netD
        }

    elif network_config.name == 'arpl_net':
        from .arpl_layer import ARPLayer
        feature_net = get_network(network_config.feat_extract_network)
        try:
            dim_centers = feature_net.fc.weight.shape[1]
            feature_net.fc = nn.Identity()
        except:
            dim_centers = feature_net.classifier[0].weight.shape[1]
            feature_net.classifier = nn.Identity()

        criterion = ARPLayer(feat_dim=dim_centers,
                             num_classes=num_classes,
                             weight_pl=network_config.weight_pl,
                             temp=network_config.temp)

        net = {'netF': feature_net, 'criterion': criterion}

    elif network_config.name == 'vgg and model':
        vgg = Vgg16(network_config['trainedsource'])
        model = make_arch(network_config['equal_network_size'],
                          network_config['use_bias'], True)
        net = {'vgg': vgg, 'model': model}

    elif network_config.name == 'bit':
        net = KNOWN_MODELS[network_config.model]()
    elif network_config.name == 'vit':
        cfg = mmcv.Config.fromfile(network_config.model)
        net = init_model(cfg, network_config.checkpoint, 0)
        net.get_fc = MethodType(
            lambda self: (self.head.layers.head.weight.cpu().numpy(),
                          self.head.layers.head.bias.cpu().numpy()), net)

    elif network_config.name == 'conf_wideresnet':
        net = Conf_WideResNet(depth=16,
                              num_classes=num_classes,
                              widen_factor=8)
    elif network_config.name == 'dcae':
        net = get_Autoencoder(network_config.type)

    elif network_config.name == 'dsvdd':
        net = build_network(network_config.type)

    elif network_config.name == 'vos':

        net = WideResNet(network_config['num_layers'],
                         num_classes,
                         network_config['widen_factor'],
                         dropRate=network_config['droprate'])
    elif network_config.name == 'projectionNet':
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
        elif network_config.name == 'bit':
            net.load_from(np.load(network_config.checkpoint))
        elif network_config.name == 'vit':
            pass
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
                net[key] = torch.nn.parallel.DistributedDataParallel(
                    subnet,
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=True)
        else:
            net = torch.nn.parallel.DistributedDataParallel(
                net.cuda(),
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=True)

    if network_config.num_gpus > 0:
        if type(net) is dict:
            for subnet in net.values():
                subnet.cuda()
        else:
            net.cuda()
        torch.cuda.manual_seed(1)
        np.random.seed(1)
    cudnn.benchmark = True
    return net
