import os
import time
from pathlib import Path

import torch

from .base_recorder import BaseRecorder


class Rd4adRecorder(BaseRecorder):
    def __init__(self, config) -> None:
        super(Rd4adRecorder, self).__init__(config)

        self.best_epoch_idx = 0
        self.best_result = 0

        self.begin_time = time.time()

    def report(self, train_metrics, test_metrics):
        print('Epoch {:03d} | Time {:5d}s | Train Loss {:.4f} | '
              'Auroc {:.4f}\n'.format(train_metrics['epoch_idx'],
                                      int(time.time() - self.begin_time),
                                      train_metrics['loss'],
                                      100.0 * test_metrics['image_auroc']),
              flush=True)

    def save_model(self, net, test_metrics):
        if self.config.recorder.save_all_models:
            torch.save(
                {
                    'bn': net['bn'].state_dict(),
                    'decoder': net['decoder'].state_dict()
                },
                os.path.join(
                    self.output_dir,
                    'model_epoch{}.ckpt'.format(test_metrics['epoch_idx'])))

        # enter only if lower loss occurs
        if test_metrics['image_auroc'] >= self.best_result:

            # delete the depreciated best model
            old_fname1 = 'bn_best_epoch{}_auroc{:.4f}.ckpt'.format(
                self.best_epoch_idx, self.best_result)
            old_fname2 = 'decoder_best_epoch{}_auroc{:.4f}.ckpt'.format(
                self.best_epoch_idx, self.best_result)

            old_pth1 = os.path.join(self.output_dir, old_fname1)
            old_pth2 = os.path.join(self.output_dir, old_fname2)
            Path(old_pth1).unlink(missing_ok=True)
            Path(old_pth2).unlink(missing_ok=True)
            # update the best model
            self.best_epoch_idx = test_metrics['epoch_idx']
            self.best_result = test_metrics['image_auroc']
            torch.save({'bn': net['bn'].state_dict()},
                       os.path.join(self.output_dir, 'bn_best.ckpt'))
            torch.save({'decoder': net['decoder'].state_dict()},
                       os.path.join(self.output_dir, 'decoder_best.ckpt'))
            save_fname1 = 'bn_best_epoch{}_auroc{:.4f}.ckpt'.format(
                self.best_epoch_idx, self.best_result)
            save_pth1 = os.path.join(self.output_dir, save_fname1)
            save_fname2 = 'decoder_best_epoch{}_auroc{:.4f}.ckpt'.format(
                self.best_epoch_idx, self.best_result)
            save_pth2 = os.path.join(self.output_dir, save_fname2)
            torch.save({'bn': net['bn'].state_dict()}, save_pth1)
            torch.save({'decoder': net['decoder'].state_dict()}, save_pth2)

    def summary(self):
        print('Training Completed!\n '
              'Best Auroc: {:.4f} at epoch {:d}\n'.format(
                  100.0 * self.best_result, self.best_epoch_idx),
              flush=True)
