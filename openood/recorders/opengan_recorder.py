import copy
import os
import time

import torch

from .base_recorder import BaseRecorder


class OpenGanRecorder(BaseRecorder):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.save_dir = self.config.output_dir
        self.G_losses = []
        self.D_losses = []
        self.D_lowest_loss = -1
        self.best_epoch_idx = 0

    def report(self, train_metrics):
        print('Epoch [{:03d}/{:03d}] | Time {:5d}s | Loss_G: {:.4f} | '
              'Loss_D: {:.4f}\n'.format(train_metrics['epoch_idx'],
                                        self.config.optimizer.num_epochs,
                                        int(time.time() - self.begin_time),
                                        train_metrics['G_losses'][-1],
                                        train_metrics['D_losses'][-1]),
              flush=True)

    def save_model(self, net, train_metrics):

        netG = net['netG']
        netD = net['netD']
        epoch_idx = train_metrics['epoch_idx']

        self.G_losses.extend(train_metrics['G_losses'])
        self.D_losses.extend(train_metrics['D_losses'])

        netG_wts = copy.deepcopy(netG.state_dict())
        netD_wts = copy.deepcopy(netD.state_dict())

        if self.config.recorder.save_all_models:
            save_pth = os.path.join(self.save_dir,
                                    'epoch-{}_GNet.ckpt'.format(epoch_idx))
            torch.save(netG_wts, save_pth)
            save_pth = os.path.join(self.save_dir,
                                    'epoch-{}_DNet.ckpt'.format(epoch_idx))
            torch.save(netD_wts, save_pth)

        if self.D_lowest_loss == -1 or self.D_losses[-1] <= self.D_lowest_loss:
            # # delete the depreciated best model
            # old_fname = 'best_epoch{}.ckpt'.format(
            #     self.best_epoch_idx)
            # old_pth = os.path.join(self.output_dir, old_fname)
            # Path(old_pth).unlink(missing_ok=True)

            self.best_epoch_idx = epoch_idx
            self.D_lowest_loss = self.D_losses[-1]

            torch.save(netG_wts, os.path.join(self.output_dir,
                                              'best_GNet.ckpt'))
            torch.save(netD_wts, os.path.join(self.output_dir,
                                              'best_DNet.ckpt'))

    def summary(self):
        print('Training Completed! '
              'Lowest loss on netD: {:.6f} '
              'at epoch {:d}'.format(self.D_lowest_loss, self.best_epoch_idx),
              flush=True)
