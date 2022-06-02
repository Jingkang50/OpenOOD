import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.utils import Config

from .lr_scheduler import cosine_annealing


class ARPLGANTrainer:
    def __init__(self, net: dict, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net['netF']
        self.netG = net['netG']
        self.netD = net['netD']
        self.train_loader = train_loader
        self.config = config
        self.criterion = net['criterion']

        self.fixed_noise = torch.FloatTensor(64, config.network.nz, 1,
                                             1).normal_(0, 1).cuda()
        self.criterionD = nn.BCELoss()

        params_list = [{
            'params': self.net.parameters()
        }, {
            'params': self.criterion.parameters()
        }]

        self.optimizer = torch.optim.SGD(
            params_list,
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )

        self.optimizerD = torch.optim.Adam(self.netD.parameters(),
                                           lr=config.optimizer.gan_lr,
                                           betas=(0.5, 0.999))
        self.optimizerG = torch.optim.Adam(self.netG.parameters(),
                                           lr=config.optimizer.gan_lr,
                                           betas=(0.5, 0.999))

    def train_epoch(self, epoch_idx):
        self.net.train()
        self.netD.train()
        self.netG.train()

        loss_avg, lossG_avg, lossD_avg = 0.0, 0.0, 0.0
        train_dataiter = iter(self.train_loader)

        real_label, fake_label = 1, 0
        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()
            gan_target = torch.FloatTensor(target.size()).fill_(0).cuda()

            noise = torch.FloatTensor(
                data.size(0), self.config.network.nz, self.config.network.ns,
                self.config.network.ns).normal_(0, 1).cuda()
            noise = noise.cuda()
            noise = Variable(noise)
            fake = self.netG(noise)

            ###########################
            # (1) Update D network    #
            ###########################
            # train with real
            gan_target.fill_(real_label)
            targetv = Variable(gan_target)
            self.optimizerD.zero_grad()
            output = self.netD(data)
            errD_real = self.criterionD(output, targetv)
            errD_real.backward()

            # train with fake
            targetv = Variable(gan_target.fill_(fake_label))
            output = self.netD(fake.detach())
            errD_fake = self.criterionD(output, targetv)
            errD_fake.backward()
            errD = errD_real + errD_fake
            self.optimizerD.step()

            ###########################
            # (2) Update G network    #
            ###########################
            self.optimizerG.zero_grad()
            # Original GAN loss
            targetv = Variable(gan_target.fill_(real_label))
            output = self.netD(fake)
            errG = self.criterionD(output, targetv)

            # minimize the true distribution
            _, feat = self.net(
                fake, True,
                1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
            errG_F = self.criterion.fake_loss(feat).mean()
            generator_loss = errG + self.config.loss.beta * errG_F
            generator_loss.backward()
            self.optimizerG.step()

            ###########################
            # (3) Update classifier   #
            ###########################
            # cross entropy loss
            self.optimizer.zero_grad()
            _, feat = self.net(
                data, True,
                0 * torch.ones(data.shape[0], dtype=torch.long).cuda())
            _, loss = self.criterion(feat, target)

            # KL divergence
            noise = torch.FloatTensor(
                data.size(0), self.config.network.nz, self.config.network.ns,
                self.config.network.ns).normal_(0, 1).cuda()
            noise = Variable(noise)
            fake = self.netG(noise)
            _, feat = self.net(
                fake, True,
                1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
            F_loss_fake = self.criterion.fake_loss(feat).mean()
            total_loss = loss + self.config.loss.beta * F_loss_fake
            total_loss.backward()
            self.optimizer.step()

            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(total_loss) * 0.2
                lossG_avg = lossG_avg * 0.8 + float(generator_loss) * 0.2
                lossD_avg = lossD_avg * 0.8 + float(errD) * 0.2

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_avg
        metrics['lossG'] = lossG_avg
        metrics['lossD'] = lossD_avg

        return {
            'netG': self.netG,
            'netD': self.netD,
            'netF': self.net,
            'criterion': self.criterion
        }, metrics
