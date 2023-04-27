import os
import time
from pathlib import Path

import torch


class CiderRecorder:
    def __init__(self, config) -> None:
        self.config = config

        self.best_loss = float('inf')
        self.best_epoch_idx = 0

        self.begin_time = time.time()
        self.output_dir = config.output_dir

    def report(self, train_metrics, val_metrics):
        print('\nEpoch {:03d} | Time {:5d}s | Train Loss {:.4f}'.format(
            (train_metrics['epoch_idx']), int(time.time() - self.begin_time),
            train_metrics['loss']),
              flush=True)

    def save_model(self, net, train_metrics):
        try:
            state_dict = net.module.state_dict()
        except AttributeError:
            state_dict = net.state_dict()

        if self.config.recorder.save_all_models:
            torch.save(
                state_dict,
                os.path.join(
                    self.output_dir,
                    'model_epoch{}.ckpt'.format(train_metrics['epoch_idx'])))

        # enter only if better accuracy occurs
        if train_metrics['loss'] <= self.best_loss:
            # delete the depreciated best model
            old_fname = 'best_epoch{}_loss{:.4f}.ckpt'.format(
                self.best_epoch_idx, self.best_loss)
            old_pth = os.path.join(self.output_dir, old_fname)
            Path(old_pth).unlink(missing_ok=True)

            # update the best model
            self.best_epoch_idx = train_metrics['epoch_idx']
            self.best_loss = train_metrics['loss']
            torch.save(state_dict, os.path.join(self.output_dir, 'best.ckpt'))

            save_fname = 'best_epoch{}_loss{:.4f}.ckpt'.format(
                self.best_epoch_idx, self.best_loss)
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(state_dict, save_pth)

        # save last path
        if train_metrics['epoch_idx'] == self.config.optimizer.num_epochs:
            save_fname = 'last_epoch{}_loss{:.4f}.ckpt'.format(
                train_metrics['epoch_idx'], train_metrics['loss'])
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(state_dict, save_pth)

    def summary(self):
        print('Training Completed! '
              'Best loss: {:.4f} '
              'at epoch {:d}'.format(self.best_loss, self.best_epoch_idx),
              flush=True)
