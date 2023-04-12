import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import openood.utils.comm as comm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class OpenGanTrainer:
    def __init__(self, net, feat_loader, config) -> None:

        manualSeed = 999
        print('Random Seed: ', manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        self.config = config
        self.netG = net['netG']
        self.netD = net['netD']
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        self.feat_loader = feat_loader

        self.nz = self.config.network.nz

        self.real_label = 1
        self.fake_label = 0

        optimizer_config = self.config.optimizer
        self.optimizerD = optim.Adam(self.netD.parameters(),
                                     lr=optimizer_config.lr / 1.5,
                                     betas=(optimizer_config.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(),
                                     lr=optimizer_config.lr,
                                     betas=(optimizer_config.beta1, 0.999))

        self.criterion = nn.BCELoss()

        self.G_losses = []
        self.D_losses = []

    def train_epoch(self, epoch_idx):

        feat_dataiter = iter(self.feat_loader)

        for train_step in tqdm(range(1,
                                     len(feat_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            data = next(feat_dataiter)['data']
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            self.netD.zero_grad()
            # Format batch
            loaded_data = data.cuda()
            b_size = loaded_data.size(0)
            label = torch.full((b_size, ), self.real_label).cuda()
            label = label.to(torch.float32)

            # Forward pass real batch through D
            output = self.netD(loaded_data).view(-1)
            # import pdb
            # pdb.set_trace()
            # Calculate loss on all-real batch
            errD_real = self.criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, self.nz, 1, 1).cuda()
            # Generate fake image batch with G
            fake = self.netG(noise)
            label.fill_(self.fake_label)
            # Classify all fake batch with D
            output = self.netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = self.criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            self.optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.netG.zero_grad()
            label.fill_(
                self.real_label)  # fake labels are real for generator cost
            # Since we just updated D,
            # perform another forward pass of all-fake batch through D
            output = self.netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = self.criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            self.optimizerG.step()

            # Save Losses for plotting later, if needed
            self.G_losses.append(errG.item())
            self.D_losses.append(errD.item())

        return {
            'netG': self.netG,
            'netD': self.netD
        }, {
            'G_losses': self.G_losses,
            'D_losses': self.D_losses,
            'epoch_idx': epoch_idx
        }
