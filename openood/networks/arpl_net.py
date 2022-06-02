## reference code https://github.com/pytorch/examples/blob/master/dcgan/main.py

import operator
from collections import OrderedDict
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netD32(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netD32, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input size. (nc) x 32 x 32
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 1, 0, bias=False),
            nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(ndf * 16, 1), nn.Sigmoid())

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.classifier(output).flatten()

        return output


class _netG32(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(_netG32, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            # nn.Sigmoid()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        return output


def Generator32(n_gpu, nz, ngf, nc):
    model = _netG32(n_gpu, nz, ngf, nc)
    model.apply(weights_init)
    return model


def Discriminator32(n_gpu, nc, ndf):
    model = _netD32(n_gpu, nc, ndf)
    model.apply(weights_init)
    return model


class _netD(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input size. (nc) x 32 x 32
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 1, 0, bias=False),
            nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(ndf * 16, 1), nn.Sigmoid())

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.classifier(output).flatten()

        return output


class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        return output


def Generator(n_gpu, nz, ngf, nc):
    model = _netG(n_gpu, nz, ngf, nc)
    model.apply(weights_init)
    return model


def Discriminator(n_gpu, nc, ndf):
    model = _netD(n_gpu, nc, ndf)
    model.apply(weights_init)
    return model


class _MultiBatchNorm(nn.Module):
    _version = 2

    def __init__(self,
                 num_features,
                 num_classes,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(_MultiBatchNorm, self).__init__()
        # self.bns = nn.ModuleList([nn.modules.batchnorm._BatchNorm(
        # num_features, eps, momentum, affine, track_running_stats)
        # for _ in range(num_classes)])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps, momentum, affine,
                           track_running_stats) for _ in range(num_classes)
        ])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label[0]]
        return bn(x), domain_label


class MultiBatchNorm(_MultiBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                input.dim()))


_pair = _ntuple(2)

__all__ = [
    'resnet18ABN', 'resnet34ABN', 'resnet50ABN', 'resnet101ABN', 'resnet152ABN'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class Conv2d(_ConvNd):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride,
                                     padding,
                                     dilation,
                                     False,
                                     _pair(0),
                                     groups,
                                     bias,
                                     padding_mode='zeros')

    def forward(self, input, domain_label):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups), domain_label


class TwoInputSequential(nn.Module):
    r"""A sequential container forward with two inputs.
    """
    def __init__(self, *args):
        super(TwoInputSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator."""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TwoInputSequential(
                OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(TwoInputSequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input1, input2):
        for module in self._modules.values():
            input1, input2 = module(input1, input2)
        return input1, input2


def resnet18ABN(num_classes=10, num_bns=2):
    model = ResNetABN(BasicBlock, [2, 2, 2, 2],
                      num_classes=num_classes,
                      num_bns=num_bns)

    return model


def resnet34ABN(num_classes=10, num_bns=2):
    model = ResNetABN(BasicBlock, [3, 4, 6, 3],
                      num_classes=num_classes,
                      num_bns=num_bns)

    return model


def resnet50ABN(num_classes=10, num_bns=2):
    model = ResNetABN(Bottleneck, [3, 4, 6, 3],
                      num_classes=num_classes,
                      num_bns=num_bns)

    return model


def _update_initial_weights_ABN(state_dict,
                                num_classes=1000,
                                num_bns=2,
                                ABN_type='all'):
    new_state_dict = state_dict.copy()

    for key, val in state_dict.items():
        update_dict = False
        if ((('bn' in key or 'downsample.1' in key) and ABN_type == 'all')
                or (('bn1' in key) and ABN_type == 'partial-bn1')):
            update_dict = True

        if (update_dict):
            if 'weight' in key:
                for d in range(num_bns):
                    new_state_dict[
                        key[0:-6] +
                        'bns.{}.weight'.format(d)] = val.data.clone()

            elif 'bias' in key:
                for d in range(num_bns):
                    new_state_dict[key[0:-4] +
                                   'bns.{}.bias'.format(d)] = val.data.clone()

            if 'running_mean' in key:
                for d in range(num_bns):
                    new_state_dict[
                        key[0:-12] +
                        'bns.{}.running_mean'.format(d)] = val.data.clone()

            if 'running_var' in key:
                for d in range(num_bns):
                    new_state_dict[
                        key[0:-11] +
                        'bns.{}.running_var'.format(d)] = val.data.clone()

            if 'num_batches_tracked' in key:
                for d in range(num_bns):
                    new_state_dict[key[0:-len('num_batches_tracked')] +
                                   'bns.{}.num_batches_tracked'.format(
                                       d)] = val.data.clone()

    if num_classes != 1000 or len(
        [key for key in new_state_dict.keys() if 'fc' in key]) > 1:
        key_list = list(new_state_dict.keys())
        for key in key_list:
            if 'fc' in key:
                print('pretrained {} are not used as initial params.'.format(
                    key))
                del new_state_dict[key]

    return new_state_dict


class ResNetABN(nn.Module):
    def __init__(self, block, layers, num_classes=10, num_bns=2):
        self.inplanes = 64
        self.num_bns = num_bns
        self.num_classes = num_classes
        super(ResNetABN, self).__init__()
        self.conv1 = conv3x3(3, 64)
        self.bn1 = MultiBatchNorm(64, self.num_bns)
        self.layer1 = self._make_layer(block,
                                       64,
                                       layers[0],
                                       stride=1,
                                       num_bns=self.num_bns)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       num_bns=self.num_bns)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       num_bns=self.num_bns)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       num_bns=self.num_bns)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, num_bns=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = TwoInputSequential(
                Conv2d(self.inplanes,
                       planes * block.expansion,
                       kernel_size=1,
                       stride=stride,
                       bias=False),
                MultiBatchNorm(planes * block.expansion, num_bns),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, num_bns=num_bns))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_bns=num_bns))

        return TwoInputSequential(*layers)

    def forward(self, x, return_feature=False, domain_label=None):
        if domain_label is None:
            domain_label = 0 * torch.ones(x.shape[0], dtype=torch.long).cuda()
        x = self.conv1(x)
        x, _ = self.bn1(x, domain_label)
        x = F.relu(x)
        x, _ = self.layer1(x, domain_label)
        x, _ = self.layer2(x, domain_label)
        x, _ = self.layer3(x, domain_label)
        x, _ = self.layer4(x, domain_label)

        x = self.avgpool(x)
        feat = x.view(x.size(0), -1)
        x = self.fc(feat)

        if return_feature:
            return x, feat
        else:
            return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_bns=2):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = MultiBatchNorm(planes, num_bns)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = MultiBatchNorm(planes, num_bns)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, domain_label):
        residual = x

        out = self.conv1(x)
        out, _ = self.bn1(out, domain_label)
        out = F.relu(out)

        out = self.conv2(out)
        out, _ = self.bn2(out, domain_label)

        if self.downsample is not None:
            residual, _ = self.downsample(x, domain_label)

        out += residual
        out = F.relu(out)

        return out, domain_label


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_bns=2):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = MultiBatchNorm(planes, num_bns)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = MultiBatchNorm(planes, num_bns)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = MultiBatchNorm(planes * 4, num_bns)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, domain_label):
        residual = x

        out = self.conv1(x)
        out, _ = self.bn1(out, domain_label)
        out = self.relu(out)

        out = self.conv2(out)
        out, _ = self.bn2(out, domain_label)
        out = self.relu(out)

        out = self.conv3(out)
        out, _ = self.bn3(out, domain_label)

        if self.downsample is not None:
            residual, _ = self.downsample(x, domain_label)

        out += residual
        out = self.relu(out)

        return out, domain_label


class Dist(nn.Module):
    def __init__(self,
                 num_classes=10,
                 num_centers=1,
                 feat_dim=2,
                 init='random'):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers

        if init == 'random':
            self.centers = nn.Parameter(
                0.1 * torch.randn(num_classes * num_centers, self.feat_dim))
        else:
            self.centers = nn.Parameter(
                torch.Tensor(num_classes * num_centers, self.feat_dim))
            self.centers.data.fill_(0)

    def forward(self, features, center=None, metric='l2'):
        if metric == 'l2':
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
            if center is None:
                c_2 = torch.sum(torch.pow(self.centers, 2),
                                dim=1,
                                keepdim=True)
                dist = f_2 - 2 * torch.matmul(
                    features, torch.transpose(self.centers, 1,
                                              0)) + torch.transpose(c_2, 1, 0)
            else:
                c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
                dist = f_2 - 2 * torch.matmul(
                    features, torch.transpose(center, 1, 0)) + torch.transpose(
                        c_2, 1, 0)
            dist = dist / float(features.shape[1])
        else:
            if center is None:
                center = self.centers
            else:
                center = center
            dist = features.matmul(center.t())
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        dist = torch.mean(dist, dim=2)

        return dist


class ARPLayer(nn.Module):
    def __init__(self, feat_dim=2, num_classes=10, weight_pl=0.1, temp=1.0):
        super(ARPLayer, self).__init__()
        self.weight_pl = weight_pl
        self.temp = temp
        self.Dist = Dist(num_classes, feat_dim=feat_dim)
        self.points = self.Dist.centers
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

    def forward(self, x, labels=None):
        dist_dot_p = self.Dist(x, center=self.points, metric='dot')
        dist_l2_p = self.Dist(x, center=self.points)
        logits = dist_l2_p - dist_dot_p

        if labels is None: return logits
        loss = F.cross_entropy(logits / self.temp, labels)

        center_batch = self.points[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).cuda()
        loss_r = self.margin_loss(self.radius, _dis_known, target)

        loss = loss + self.weight_pl * loss_r

        return logits, loss

    def fake_loss(self, x):
        logits = self.Dist(x, center=self.points)
        prob = F.softmax(logits, dim=1)
        loss = (prob * torch.log(prob)).sum(1).mean().exp()

        return loss
