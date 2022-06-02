import os
from pathlib import Path

import torch

from .ad_recorder import ADRecorder


class DRAEMRecorder(ADRecorder):
    def __init__(self, config) -> None:
        super(DRAEMRecorder, self).__init__(config)

        self.best_model_basis = self.config.recorder.best_model_basis

        self.run_name = ('draem_test_' + str(self.config.optimizer.lr) + '_' +
                         str(self.config.optimizer.num_epochs) + '_bs' +
                         str(self.config.dataset.train.batch_size) + '_' +
                         self.config.dataset.name)

    def save_model(self, net, test_metrics):
        if self.config.recorder.save_all_models:

            save_fname = self.run_name + '_model_epoch{}'.format(
                test_metrics['epoch_idx'])
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(net['generative'].state_dict(), save_pth + '.ckpt')
            torch.save(net['discriminative'].state_dict(),
                       save_pth + '_seg.ckpt')

        # enter only if lower loss occurs
        if test_metrics[self.best_model_basis] >= self.best_result:

            # delete the depreciated best model
            old_fname = self.run_name + '_best_epoch{}_loss{:.4f}'.format(
                self.best_epoch_idx, self.best_result)
            old_pth = os.path.join(self.output_dir, old_fname)
            Path(old_pth + '.ckpt').unlink(missing_ok=True)
            Path(old_pth + '_seg.ckpt').unlink(missing_ok=True)

            # update the best model
            self.best_epoch_idx = test_metrics['epoch_idx']
            self.best_result = test_metrics[self.best_model_basis]

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
