from torch import nn


class Generator(nn.Module):
    def __init__(self, in_channels=100, feature_size=64, out_channels=512):
        super(Generator, self).__init__()
        self.nz = in_channels
        self.ngf = feature_size
        self.nc = out_channels

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # Conv2d(in_channels,
            #        out_channels,
            #        kernel_size,
            #        stride=1,
            #        padding=0,
            #        dilation=1,
            #        groups=1,
            #        bias=True,
            #        padding_mode='zeros')
            nn.Conv2d(self.nz, self.ngf * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (self.ngf*8) x 4 x 4
            nn.Conv2d(self.ngf * 8, self.ngf * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (self.ngf*4) x 8 x 8
            nn.Conv2d(self.ngf * 4, self.ngf * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (self.ngf*2) x 16 x 16
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (self.ngf) x 32 x 32
            nn.Conv2d(self.ngf * 4, self.nc, 1, 1, 0, bias=True),
            # nn.Tanh()
            # state size. (self.nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, in_channels=512, feature_size=64):
        super(Discriminator, self).__init__()
        self.nc = in_channels
        self.ndf = feature_size
        self.main = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf * 8, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 8, self.ndf * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 4, self.ndf * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 2, self.ndf, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, 1, 1, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, input):
        return self.main(input)
