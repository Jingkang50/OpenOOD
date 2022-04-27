import os
import time
from pathlib import Path

import torch

from .base_recorder import BaseRecorder


class DRAEMRecorder(BaseRecorder):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.best_model_basis = self.config.recorder.best_model_basis

        self.best = {
            'image_auc': {
                'value': 0.0,
                'epoch_idx': 0
            },
            'image_ap': {
                'value': 0.0,
                'epoch_idx': 0
            },
            'pixel_auc': {
                'value': 0.0,
                'epoch_idx': 0
            },
            'pixel_ap': {
                'value': 0.0,
                'epoch_idx': 0
            },
        }

        self.best_epoch_idx = 0
        self.best_result = -1

        self.begin_time = time.time()

        self.run_name = ('DRAEM_test_' + str(self.config.optimizer.lr) + '_' +
                         str(self.config.optimizer.num_epochs) + '_bs' +
                         str(self.config.dataset.train.batch_size) + '_' +
                         self.config.dataset.name)

    def report(self, train_metrics, test_metrics):
        print('Epoch {:03d} | Time {:5d}s | Train Loss(smoothed) {:.4f} | '
              'Best Result on {}: {:.4f}\n'.format(
                  train_metrics['epoch_idx'],
                  int(time.time() - self.begin_time),
                  train_metrics['loss_smoothed'], self.best_model_basis,
                  test_metrics[self.best_model_basis]),
              flush=True)

    def save_model(self, net, test_metrics):
        if self.config.recorder.save_all_models:

            save_fname = self.run_name + '_model_epoch{}'.format(
                test_metrics['epoch_idx'])
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(net['generative'].state_dict(), save_pth + '.ckpt')
            torch.save(net['discriminative'].state_dict(),
                       save_pth + '_seg.ckpt')

        # enter only if lower loss occurs
        if self.best_result == -1 or test_metrics[
                self.best_model_basis] >= self.best_result:

            # delete the depreciated best model
            old_fname = self.run_name + '_best_epoch{}_loss{:.4f}'.format(
                self.best_epoch_idx, self.best_result)
            old_pth = os.path.join(self.output_dir, old_fname)
            Path(old_pth + '.ckpt').unlink(missing_ok=True)
            Path(old_pth + '_seg.ckpt').unlink(missing_ok=True)

            # update the best model
            self.best_epoch_idx = test_metrics['epoch_idx']
            self.best_result = test_metrics[self.best_model_basis]

            # torch.save(net['generative'].state_dict(),
            #            os.path.join(self.output_dir, 'best.ckpt'))
            # torch.save(net['discriminative'].state_dict(),
            #            os.path.join(self.output_dir, 'best_seg.ckpt'))

            save_fname = self.run_name + '_best_epoch{}_loss{:.4f}'.format(
                self.best_epoch_idx, self.best_result)
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(net['generative'].state_dict(), save_pth + '.ckpt')
            torch.save(net['discriminative'].state_dict(),
                       save_pth + '_seg.ckpt')

        if test_metrics['epoch_idx'] == self.config.optimizer.num_epochs:
            save_fname = self.run_name + '_latest_checkpoint'
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(net['generative'].state_dict(), save_pth + '.ckpt')
            torch.save(net['discriminative'].state_dict(),
                       save_pth + '_seg.ckpt')

    def summary(self):
        print(
            'Training Completed!\n '
            #   'Best AUC Image: {:.2f} at epoch {:d}\n'
            #   'Best AP Image: {:.2f} at epoch {:d}\n'
            #   'Best AUC Pixel: {:.2f} at epoch {:d}\n'
            #   'Best AUC Pixel: {:.2f} at epoch {:d}\n'.format(
            #       100 * self.best['image_auc']['value'],
            #       self.best['image_auc']['epoch_idx'],
            #       100 * self.best['image_ap']['value'],
            #       self.best['image_ap']['epoch_idx'],
            #       100 * self.best['pixel_auc']['value'],
            #       self.best['pixel_auc']['epoch_idx'],
            #       100 * self.best['pixel_ap']['value'],
            #       self.best['pixel_ap']['epoch_idx'])
            'lowest loss: {:.4f} at epoch {:d}\n'.format(
                self.best_result, self.best_epoch_idx),
            flush=True)
