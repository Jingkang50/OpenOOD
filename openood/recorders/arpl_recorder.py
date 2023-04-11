import copy
import os
import time

import torch

from .base_recorder import BaseRecorder


class ARPLRecorder(BaseRecorder):
    def __init__(self, config) -> None:
        super().__init__(config)

    def report(self, train_metrics, val_metrics):
        if 'lossD' in train_metrics.keys():
            print('\nEpoch {:03d} | Time {:5d}s | D Loss {:.4f} | '
                  'G Loss {:.4f} | Train Loss {:.4f} | '
                  'Val Loss {:.3f} | Val Acc {:.2f}'.format(
                      (train_metrics['epoch_idx']),
                      int(time.time() - self.begin_time),
                      train_metrics['lossD'], train_metrics['lossG'],
                      train_metrics['loss'], val_metrics['loss'],
                      100.0 * val_metrics['acc']),
                  flush=True)
        else:
            print('\nEpoch {:03d} | Time {:5d}s | Train Loss {:.4f} | '
                  'Val Loss {:.3f} | Val Acc {:.2f}'.format(
                      (train_metrics['epoch_idx']),
                      int(time.time() - self.begin_time),
                      train_metrics['loss'], val_metrics['loss'],
                      100.0 * val_metrics['acc']),
                  flush=True)

    def save_model(self, net, val_metrics):

        netF = net['netF']
        criterion = net['criterion']
        epoch_idx = val_metrics['epoch_idx']

        try:
            netF_wts = copy.deepcopy(netF.module.state_dict())
            criterion_wts = copy.deepcopy(criterion.module.state_dict())
        except AttributeError:
            netF_wts = copy.deepcopy(netF.state_dict())
            criterion_wts = copy.deepcopy(criterion.state_dict())

        if self.config.recorder.save_all_models:
            save_pth = os.path.join(self.save_dir,
                                    'epoch-{}_NetF.ckpt'.format(epoch_idx))
            torch.save(netF_wts, save_pth)
            save_pth = os.path.join(
                self.save_dir, 'epoch-{}_criterion.ckpt'.format(epoch_idx))
            torch.save(criterion_wts, save_pth)

        # enter only if better accuracy occurs
        if val_metrics['acc'] >= self.best_acc:

            # update the best model
            self.best_epoch_idx = val_metrics['epoch_idx']
            self.best_acc = val_metrics['acc']

            torch.save(netF_wts, os.path.join(self.output_dir,
                                              'best_NetF.ckpt'))
            torch.save(criterion_wts,
                       os.path.join(self.output_dir, 'best_criterion.ckpt'))
