import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.losses.draem_loss import get_draem_losses
from openood.utils import Config


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class DRAEMTrainer:
    def __init__(self, net, train_loader: DataLoader, config: Config) -> None:
        self.config = config
        self.net = net
        self.net['generative'].apply(weights_init)
        self.net['discriminative'].apply(weights_init)
        self.train_loader = train_loader

        self.optimizer = torch.optim.Adam([{
            'params':
            self.net['generative'].parameters(),
            'lr':
            self.config.optimizer.lr
        }, {
            'params':
            self.net['discriminative'].parameters(),
            'lr':
            self.config.optimizer.lr
        }])

        steps = []
        for step in self.config.optimizer.steps:
            steps.append(self.config.optimizer.num_epochs * step)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        steps,
                                                        gamma=0.2,
                                                        last_epoch=-1)

        self.losses = get_draem_losses()

    def train_epoch(self, epoch_idx):
        self.net['generative'].train()
        self.net['discriminative'].train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True):
            sample_batched = next(train_dataiter)
            gray_batch = sample_batched['data']['image'].cuda()
            aug_gray_batch = sample_batched['data']['augmented_image'].cuda()
            anomaly_mask = sample_batched['data']['anomaly_mask'].cuda()

            # forward
            gray_rec = self.net['generative'](aug_gray_batch)
            # conconcat origin and generated
            joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

            out_mask = self.net['discriminative'](joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            l2_loss = self.losses['l2'](gray_rec, gray_batch)
            ssim_loss = self.losses['ssim'](gray_rec, gray_batch)

            segment_loss = self.losses['focal'](out_mask_sm, anomaly_mask)
            loss = l2_loss + ssim_loss + segment_loss

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        self.scheduler.step()
        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss_smoothed'] = loss_avg
        metrics['loss'] = loss

        return self.net, metrics
