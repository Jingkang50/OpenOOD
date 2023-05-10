import copy
import os
import time

import torch

from .base_recorder import BaseRecorder


class OpenGanRecorder(BaseRecorder):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.save_dir = self.config.output_dir
        self.best_val_auroc = 0
        self.best_epoch_idx = 0

    def report(self, train_metrics, val_metrics):
        print('Epoch [{:03d}/{:03d}] | Time {:5d}s | Loss_G: {:.4f} | '
              'Loss_D: {:.4f} | Val AUROC: {:.2f}\n'.format(
                  train_metrics['epoch_idx'], self.config.optimizer.num_epochs,
                  int(time.time() - self.begin_time),
                  train_metrics['G_losses'][-1], train_metrics['D_losses'][-1],
                  val_metrics['auroc']),
              flush=True)

    def save_model(self, net, val_metrics):
        netG = net['netG']
        netD = net['netD']
        epoch_idx = val_metrics['epoch_idx']

        try:
            netG_wts = copy.deepcopy(netG.module.state_dict())
            netD_wts = copy.deepcopy(netD.module.state_dict())
        except AttributeError:
            netG_wts = copy.deepcopy(netG.state_dict())
            netD_wts = copy.deepcopy(netD.state_dict())

        if self.config.recorder.save_all_models:
            save_pth = os.path.join(self.save_dir,
                                    'epoch-{}_GNet.ckpt'.format(epoch_idx))
            torch.save(netG_wts, save_pth)
            save_pth = os.path.join(self.save_dir,
                                    'epoch-{}_DNet.ckpt'.format(epoch_idx))
            torch.save(netD_wts, save_pth)

        if val_metrics['auroc'] >= self.best_val_auroc:
            self.best_epoch_idx = epoch_idx
            self.best_val_auroc = val_metrics['auroc']

            torch.save(netG_wts, os.path.join(self.output_dir,
                                              'best_GNet.ckpt'))
            torch.save(netD_wts, os.path.join(self.output_dir,
                                              'best_DNet.ckpt'))

    def summary(self):
        print('Training Completed! '
              'Best val AUROC on netD: {:.6f} '
              'at epoch {:d}'.format(self.best_val_auroc, self.best_epoch_idx),
              flush=True)
