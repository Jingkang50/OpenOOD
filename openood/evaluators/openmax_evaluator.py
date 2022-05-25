import csv
import os
from typing import Dict, List

import numpy as np
import torch.nn as nn
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader

from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .base_evaluator import BaseEvaluator
from .metrics import compute_all_metrics


class OpenMaxEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        super(OpenMaxEvaluator, self).__init__(config)

    def eval_ood(self, net: nn.Module, id_data_loader: DataLoader,
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor: BasePostprocessor):
        net.eval()
        # load training in-distribution data
        assert 'test' in id_data_loader, \
            'id_data_loaders should have the key: test!'
        dataset_name = self.config.dataset.name
        print(f'Performing inference on {dataset_name} dataset...', flush=True)
        id_pred, id_conf, id_gt = postprocessor.inference(
            net, id_data_loader['test'])

        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='nearood')
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='farood')

    def _eval_ood(self,
                  net: nn.Module,
                  id_list: List[np.ndarray],
                  ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                  postprocessor: BasePostprocessor,
                  ood_split: str = 'nearood'):
        print(f'Processing {ood_split}...', flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        for dataset_name, ood_dl in ood_data_loaders[ood_split].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            ood_pred, ood_conf, ood_gt = postprocessor.inference(net, ood_dl)
            

            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            self.label = label
            self.predict = pred

            print(f'Computing metrics on {dataset_name} dataset...')

            print(f'OpenMax F1 is ')
            print(f1_score(label, pred, average='micro'))

            print(f'OpenMax f1_macro is ')
            print(f1_score(label, pred, average='macro'))

            print(f'OpenMax f1_macro_weighted is')
            print(f1_score(label, pred, average='weighted'))

            print(f'OpenMax area_under_roc is')
            print(self._area_under_roc(conf))
            print(u'\u2500' * 70, flush=True)

    def _area_under_roc(self,
                        prediction_scores: np.array = None,
                        multi_class='ovo') -> float:
        """Area Under Receiver Operating Characteristic Curve.

        :param prediction_scores: array-like of shape (n_samples, n_classes). The multi-class ROC curve requires
            prediction scores for each class. If not specified, will generate its own prediction scores that assume
            100% confidence in selected prediction.
        :param multi_class: {'ovo', 'ovr'}, default='ovo'
            'ovo' computes the average AUC of all possible pairwise combinations of classes.
            'ovr' Computes the AUC of each class against the rest.
        :return: float representing the area under the ROC curve
        """
        label, predict = self.label, self.predict
        one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        one_hot_encoder.fit(np.array(label).reshape(-1, 1))
        true_scores = one_hot_encoder.transform(np.array(label).reshape(-1, 1))
        if prediction_scores is None:
            prediction_scores = one_hot_encoder.transform(
                np.array(predict).reshape(-1, 1))
        
        # assert prediction_scores.shape == true_scores.shape
        return roc_auc_score(true_scores,
                             prediction_scores,
                             multi_class=multi_class)

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        net.eval()
        id_pred, _, id_gt = postprocessor.inference(net, data_loader)
        metrics = {}
        metrics['acc'] = sum(id_pred == id_gt) / len(id_pred)
        metrics['epoch_idx'] = epoch_idx
        return metrics

    def report(self, test_metrics):
        print('Completed!', flush=True)
