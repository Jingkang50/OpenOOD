import os
import time
from pathlib import Path

import torch


class CutpasteRecorder:
    def __init__(self, config) -> None:
        self.config = config

        self.best_auroc = 0.0
        self.best_epoch_idx = 0

        self.begin_time = time.time()
        self.output_dir = config.output_dir

    def report(self, train_metrics, val_metrics):
        print('\nEpoch {:03d} | Time {:5d}s | Train Loss {:.4f} | '
              'AUROC {:.3f}'.format((val_metrics['epoch_idx']),
                                    int(time.time() - self.begin_time),
                                    train_metrics['loss'],
                                    val_metrics['image_auroc']),
              flush=True)

    def save_model(self, net, val_metrics):
        if self.config.recorder.save_all_models:
            torch.save(
                net.state_dict(),
                os.path.join(
                    self.output_dir,
                    'model_epoch{}.ckpt'.format(val_metrics['epoch_idx'])))

        # enter only if best auroc occurs
        if val_metrics['image_auroc'] >= self.best_auroc:

            # delete the depreciated best model
            old_fname = 'best_epoch{}_auroc{}.ckpt'.format(
                self.best_epoch_idx, self.best_auroc)
            old_pth = os.path.join(self.output_dir, old_fname)
            Path(old_pth).unlink(missing_ok=True)

            # update the best model
            self.best_epoch_idx = val_metrics['epoch_idx']
            self.best_auroc = val_metrics['image_auroc']
            torch.save(net.state_dict(),
                       os.path.join(self.output_dir, 'best.ckpt'))

            save_fname = 'best_epoch{}_auroc{}.ckpt'.format(
                self.best_epoch_idx, self.best_auroc)
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(net.state_dict(), save_pth)

    def summary(self):
        print('Training Completed! '
              'Best auroc: {:.2f} '
              'at epoch {:d}'.format(self.best_auroc, self.best_epoch_idx),
              flush=True)
