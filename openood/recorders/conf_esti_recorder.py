import os
import time
from pathlib import Path

import torch


class Conf_Esti_Recorder:
    def __init__(self, config) -> None:
        self.config = config

        self.best_acc = 0.0
        self.best_epoch_idx = 0

        self.begin_time = time.time()
        self.output_dir = config.output_dir

    def report(self, train_metrics, val_metrics):
        print(
            'Epoch {:03d} | Time {:5d}s | Train Acc {:.3f}| Val Acc {:.3f} |'.
            format((train_metrics['epoch_idx']),
                   int(time.time() - self.begin_time),
                   train_metrics['train_acc'], val_metrics['acc']),
            flush=True)

    def save_model(self, net, val_metrics):
        if self.config.recorder.save_all_models:
            torch.save(
                net.state_dict(),
                os.path.join(
                    self.output_dir,
                    'model_epoch{}.ckpt'.format(val_metrics['epoch_idx'])))

        # enter only if better accuracy occurs
        if val_metrics['acc'] >= self.best_acc:

            # delete the depreciated best model
            old_fname = 'best_epoch{}_acc{}.pth'.format(
                self.best_epoch_idx, self.best_acc)
            old_pth = os.path.join(self.output_dir, old_fname)
            Path(old_pth).unlink(missing_ok=True)

            # update the best model
            self.best_epoch_idx = val_metrics['epoch_idx']
            self.best_acc = val_metrics['acc']
            torch.save(net.state_dict(),
                       os.path.join(self.output_dir, 'best.pth'))

            save_fname = 'best_epoch{}_acc{}.pth'.format(
                self.best_epoch_idx, self.best_acc)
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(net.state_dict(), save_pth)

    def summary(self):
        print('Training Completed! '
              'Best accuracy: {:.2f} '
              'at epoch {:d}'.format(100 * self.best_acc, self.best_epoch_idx),
              flush=True)
