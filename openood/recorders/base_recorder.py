import os
import time
from pathlib import Path

import torch


class BaseRecorder:
    def __init__(self, config) -> None:
        self.config = config

        self.best_acc = 0.0
        self.best_epoch_idx = 0

        self.begin_time = time.time()
        self.output_dir = config.output_dir

    def report(self, train_metrics, val_metrics):
        print('\nEpoch {:03d} | Time {:5d}s | Train Loss {:.4f} | '
              'Test Loss {:.3f} | Test Acc {:.2f}'.format(
                  (train_metrics['epoch_idx']),
                  int(time.time() - self.begin_time),
                  train_metrics['train_loss'],
                  val_metrics['test_loss'],
                  100.0 * val_metrics['test_accuracy'],
              ),
              flush=True)

    def save_model(self, net, val_metrics):
        if self.config.recorder.save_all_models:
            torch.save(
                net.state_dict(),
                os.path.join(
                    self.output_dir,
                    'model_epoch{}.ckpt'.format(val_metrics['epoch_idx'])))

        # enter only if better accuracy occurs
        if val_metrics['test_accuracy'] >= self.best_acc:

            # delete the depreciated best model
            old_fname = 'best_epoch{}_acc{}.ckpt'.format(
                self.best_epoch_idx, self.best_acc)
            old_pth = os.path.join(self.output_dir, old_fname)
            Path(old_pth).unlink(missing_ok=True)

            # update the best model
            self.best_epoch_idx = val_metrics['epoch_idx']
            self.best_acc = val_metrics['test_accuracy']
            torch.save(net.state_dict(),
                       os.path.join(self.output_dir, 'best.ckpt'))

            save_fname = 'best_epoch{}_acc{}.ckpt'.format(
                self.best_epoch_idx, self.best_acc)
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(net.state_dict(), save_pth)

    def summary(self):
        print('Training Completed! '
              'Best accuracy: {:.2f} '
              'at epoch {:d}'.format(self.best_acc, self.best_epoch_idx),
              flush=True)
