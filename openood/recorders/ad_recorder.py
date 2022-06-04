import os
import time
from pathlib import Path

import torch

from .base_recorder import BaseRecorder


class ADRecorder(BaseRecorder):
    def __init__(self, config) -> None:
        super(ADRecorder, self).__init__(config)

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
                net.state_dict(),
                os.path.join(
                    self.output_dir,
                    'model_epoch{}.ckpt'.format(test_metrics['epoch_idx'])))

        # enter only if lower loss occurs
        if test_metrics['image_auroc'] >= self.best_result:

            # delete the depreciated best model
            old_fname = 'best_epoch{}_auroc{:.4f}.ckpt'.format(
                self.best_epoch_idx, self.best_result)
            old_pth = os.path.join(self.output_dir, old_fname)
            Path(old_pth).unlink(missing_ok=True)

            # update the best model
            self.best_epoch_idx = test_metrics['epoch_idx']
            self.best_result = test_metrics['image_auroc']
            torch.save(net.state_dict(),
                       os.path.join(self.output_dir, 'best.ckpt'))

            save_fname = 'best_epoch{}_auroc{:.4f}.ckpt'.format(
                self.best_epoch_idx, self.best_result)
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(net.state_dict(), save_pth)

    def summary(self):
        print('Training Completed!\n '
              'Best Auroc: {:.4f} at epoch {:d}\n'.format(
                  100.0 * self.best_result, self.best_epoch_idx),
              flush=True)
