import time
from pickle import NONE

import torch


class BaseRecorder:
    def __init__(self, config) -> None:
        self.best_acc = 0.0
        self.best_model = None
        self.begin_epoch = time.time()
        self.output_dir = config.output_dir

    def report(self, train_metrics, val_metrics, epoch):
        print(
            'Epoch {:3d} | Time {:5d}s | Train Loss {:.4f} | '\
            'Test Loss {:.3f} | Test Acc {:.2f}'.format(
                (epoch + 1),
                int(time.time() - self.begin_epoch),
                train_metrics['train_loss'],
                val_metrics['test_loss'],
                100.0 * val_metrics['test_accuracy'],
            ),
            flush=True,
        )

    def save_best_model(self, net, val_metrics):
        if val_metrics['test_accuracy'] >= self.best_acc:
            torch.save(net.state_dict(), self.output_dir / f'best.ckpt')
            self.best_acc = val_metrics['test_accuracy']
