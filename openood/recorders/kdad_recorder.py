import os
import time
from pathlib import Path

import torch


class KdadRecorder:
    def __init__(self, config) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.best_roc_auc = 0.0
        self.best_epoch_idx = 0
        self.begin_time = time.time()

    def report(self, train_metrics, test_metrics):
        print('epoch [{}],time:{:5d}s,loss:{:.4f},roc_auc:{:.2f}'.format(
            train_metrics['epoch_idx'], int(time.time() - self.begin_time),
            train_metrics['epoch_loss'], test_metrics['roc_auc']))

    def save_model(self, net, test_metrics):
        if self.config.recorder.save_all_models:
            torch.save(
                net['model'].state_dict(),
                os.path.join(
                    self.output_dir,
                    'Clone_epoch{}.ckpt'.format(test_metrics['epoch_idx'])))

        # enter only if better accuracy occurs
        if test_metrics['roc_auc'] >= self.best_roc_auc:

            # delete the depreciated best model
            old_fname = 'Clone_best_epoch{}_roc_auc{}.pth'.format(
                self.best_epoch_idx, self.best_roc_auc)
            old_pth = os.path.join(self.output_dir, old_fname)
            Path(old_pth).unlink(missing_ok=True)

            # update the best model
            self.best_epoch_idx = test_metrics['epoch_idx']
            self.best_roc_auc = test_metrics['roc_auc']
            save_fname = 'Clone_best_epoch{}_roc_auc{}.pth'.format(
                self.best_epoch_idx, self.best_roc_auc)
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(net['model'].state_dict(), save_pth)
        if test_metrics['epoch_idx'] == self.config['last_checkpoint']:
            torch.save(
                net['model'].state_dict(),
                '{}/Cloner_{}_epoch_{}.pth'.format(self.config['output_dir'],
                                                   self.config.normal_class,
                                                   test_metrics['epoch_idx']))

    def summary(self):
        print('Training Completed! '
              'Best Roc_auc: {:.2f}%,'
              'at epoch {:d}'.format(100 * self.best_roc_auc,
                                     self.best_epoch_idx),
              flush=True)
