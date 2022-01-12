import csv
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from openood.postprocessors import BasePostprocessor

from .metrics import compute_all_metrics
from .ood_evaluator import OODEvaluator


class FSOODEvaluator(OODEvaluator):
    def eval_csid_acc(self, net: nn.Module,
                      csid_loaders: Dict[str, Dict[str, DataLoader]]):

        net.eval()
        for dataset_name, csid_dl in csid_loaders.items():
            print(f'Computing accuracy on {dataset_name} dataset...')
            correct = 0
            with torch.no_grad():
                for batch in csid_dl:
                    data = batch['data'].cuda()
                    target = batch['label'].cuda()
                    # forward
                    output = net(data)
                    # accuracy
                    pred = output.data.max(1)[1]
                    correct += pred.eq(target.data).sum().item()
            acc = correct / len(csid_dl.dataset)
            print(f'CSID set {dataset_name}: Accuracy {100 * acc:.2f}%',
                  flush=True)
            if self.config.recorder.save_csv:
                self._save_acc_results(acc, dataset_name)

    def _save_acc_results(self, acc, dataset_name):

        write_content = {
            'dataset': dataset_name,
            'FPR@95': '-',
            'AUROC': '-',
            'AUPR_IN': '-',
            'AUPR_OUT': '-',
            'CCR_4': '-',
            'CCR_3': '-',
            'CCR_2': '-',
            'CCR_1': '-',
            'ACC': '{:.2f}'.format(100 * acc),
        }
        fieldnames = list(write_content.keys())

        # print csid metric results
        print('CSID accuracy: {:.2f}'.format(100 * acc), flush=True)

        csv_path = os.path.join(self.config.output_dir, 'fsood.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def eval_ood(self, net: nn.Module, id_data_loader: List[DataLoader],
                 ood_data_loaders: List[DataLoader],
                 postprocessor: BasePostprocessor):

        net.eval()
        # load training in-distribution data
        assert 'test' in id_data_loader, \
            'id_data_loaders should have the key: test!'
        dataset_name = self.config.dataset.name
        print(f'Performing inference on {dataset_name} dataset...', flush=True)
        id_pred, id_conf, id_gt = self.inference(net, id_data_loader['test'],
                                                 postprocessor)
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        # load csid data and compute confidence
        for dataset_name, csid_dl in ood_data_loaders['csid'].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            csid_pred, csid_conf, csid_gt = self.inference(
                net, csid_dl, postprocessor)
            if self.config.recorder.save_scores:
                self._save_scores(csid_pred, csid_conf, csid_gt, dataset_name)

            id_pred = np.concatenate([id_pred, csid_pred])
            id_conf = np.concatenate([id_conf, csid_conf])
            id_gt = np.concatenate([id_gt, csid_gt])

        # load nearood data and compute ood metrics
        nearood_metrics_list = []
        for dataset_name, ood_dl in ood_data_loaders['nearood'].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            ood_pred, ood_conf, ood_gt = self.inference(
                net, ood_dl, postprocessor)
            if self.config.recorder.save_scores:
                self._save_scores(ood_pred, ood_conf, ood_gt, dataset_name)

            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            print(f'Computing metrics on {dataset_name} dataset...')
            ood_metrics = compute_all_metrics(conf, label, pred)
            if self.config.recorder.save_csv:
                self._save_csv(ood_metrics, dataset_name=dataset_name)
            nearood_metrics_list.append(ood_metrics)

        print('Computing nearood metrics...', flush=True)
        nearood_metrics_list = np.array(nearood_metrics_list)
        metrics_mean = np.mean(nearood_metrics_list, axis=0)
        if self.config.recorder.save_csv:
            self._save_csv(metrics_mean, dataset_name='nearood')

        # load farood data and compute ood metrics
        farood_metrics_list = []
        for dataset_name, ood_dl in ood_data_loaders['farood'].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            ood_pred, ood_conf, ood_gt = self.inference(
                net, ood_dl, postprocessor)
            if self.config.recorder.save_scores:
                self._save_scores(ood_pred, ood_conf, ood_gt, dataset_name)

            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            print(f'Computing metrics on {dataset_name} dataset...')
            ood_metrics = compute_all_metrics(conf, label, pred)
            if self.config.recorder.save_csv:
                self._save_csv(ood_metrics, dataset_name=dataset_name)
            nearood_metrics_list.append(ood_metrics)

        print('Computing farood metrics...', flush=True)
        farood_metrics_list = np.array(farood_metrics_list)
        metrics_mean = np.mean(farood_metrics_list, axis=0)
        if self.config.recorder.save_csv:
            self._save_csv(metrics_mean, dataset_name='farood')
