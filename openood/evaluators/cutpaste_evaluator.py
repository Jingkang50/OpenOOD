import numpy as np

from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .metrics import compute_all_metrics


class CutPasteEvaluator:
    def __init__(self, config:Config):
        self.config = config

    def report(self, test_metrics):
        print('Complete testing, AUROC:{}'.format(test_metrics['AUROC']))

    def eval_ood(self, net, id_data_loader, ood_data_loaders,
                 postprocessor: BasePostprocessor, epoch_idx: int = -1):
        net.eval()
        # load training in-distribution data
        assert 'test' in id_data_loader, \
            'id_data_loaders should have the key: test!'
        dataset_name = self.config.dataset.name
        print(f'Performing inference on {dataset_name} dataset...', flush=True)
        id_pred, id_conf, id_gt = postprocessor.inference(
            net, id_data_loader['test'])
        for idx in range(len(id_gt)):
            if id_gt[idx] == 1:
                id_gt[idx] == -1

        # load ood data and compute ood metrics
        metrics = self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='val')
        metrics['epoch_idx'] = epoch_idx
        return metrics

    def _eval_ood(self,
                  net,
                  id_list,
                  ood_data_loaders,
                  postprocessor: BasePostprocessor,
                  ood_split: str = 'val'):
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        ood_pred, ood_conf, ood_gt = postprocessor.inference(net, ood_data_loaders[ood_split])
        ood_gt = -1 * np.ones_like(ood_pred)  # hard set to -1 as ood

        pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_gt, ood_gt])

        ood_metrics = compute_all_metrics(conf, label, pred)
        metrics_list.append(ood_metrics)
        metrics = {}

        return metrics
        

