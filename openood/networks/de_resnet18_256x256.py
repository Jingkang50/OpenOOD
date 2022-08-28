import torch
import torch.nn as nn
from torch import Tensor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, upsample=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        if self.stride == 2:
            self.conv1 = nn.ConvTranspose2d(in_planes,
                                            planes,
                                            kernel_size=2,
                                            stride=stride,
                                            bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes,
                                   planes,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=1,
                                   bias=False)
        self.upsample = upsample
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            identity = self.upsample(x)
        out += identity
        out = self.relu(out)

        return out


class De_ResNet18_256x256(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=None, num_classes=10):
        super(De_ResNet18_256x256, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        self.inplanes = 512 * block.expansion
        self.layer1 = self._make_layer(block, 256, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes,
                                   planes * block.expansion,
                                   kernel_size=2,
                                   stride=stride,
                                   bias=False),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_a = self.layer1(x)  # 512*8*8->256*16*16
        feature_b = self.layer2(feature_a)  # 256*16*16->128*32*32
        feature_c = self.layer3(feature_b)  # 128*32*32->64*64*64
        return [feature_c, feature_b, feature_a]


class AttnBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample=None) -> None:
        super(AttnBasicBlock, self).__init__()

        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = norm_layer(planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class BN_layer(nn.Module):
    def __init__(
        self,
        block,
        layers: int,
        width_per_group: int = 64,
    ):
        super(BN_layer, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)

        self.conv1 = nn.Conv2d(64 * block.expansion,
                               128 * block.expansion,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn1 = self._norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128 * block.expansion,
                               256 * block.expansion,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn2 = self._norm_layer(256 * block.expansion)
        self.conv3 = nn.Conv2d(128 * block.expansion,
                               256 * block.expansion,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn3 = self._norm_layer(256 * block.expansion)

        self.conv4 = nn.Conv2d(1024 * block.expansion,
                               512 * block.expansion,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn4 = self._norm_layer(512 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes * 3,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes * 3, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        l1 = self.relu(
            self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
        l2 = self.relu(self.bn3(self.conv3(x[1])))
        feature = torch.cat([l1, l2, x[2]], 1)
        output = self.bn_layer(feature)

        return output.contiguous()
