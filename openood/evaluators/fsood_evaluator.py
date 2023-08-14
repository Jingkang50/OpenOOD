import csv
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from openood.postprocessors import BasePostprocessor

from .ood_evaluator import OODEvaluator


class FSOODEvaluator(OODEvaluator):
    def eval_csid_acc(self, net: nn.Module,
                      csid_loaders: Dict[str, Dict[str, DataLoader]]):
        # ensure the networks in eval mode
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
            if self.config.recorder.save_csv:
                self._save_acc_results(acc, dataset_name)
        print(u'\u2500' * 70, flush=True)

    def _save_acc_results(self, acc, dataset_name):
        write_content = {
            'dataset': dataset_name,
            'FPR@95': '-',
            'AUROC': '-',
            'AUPR_IN': '-',
            'AUPR_OUT': '-',
            'ACC': '{:.2f}'.format(100 * acc),
        }
        fieldnames = list(write_content.keys())
        # print csid metric results
        print('CSID[{}] accuracy: {:.2f}%'.format(dataset_name, 100 * acc),
              flush=True)
        csv_path = os.path.join(self.config.output_dir, 'csid.csv')
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
        # ensure the networks in eval mode
        net.eval()
        # load training in-distribution data
        assert 'test' in id_data_loader, \
            'id_data_loaders should have the key: test!'
        dataset_name = self.config.dataset.name
        print(f'Performing inference on {dataset_name} dataset...', flush=True)
        id_pred, id_conf, id_gt = postprocessor.inference(
            net, id_data_loader['test'])
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        # load csid data and compute confidence
        for dataset_name, csid_dl in ood_data_loaders['csid'].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            csid_pred, csid_conf, csid_gt = postprocessor.inference(
                net, csid_dl)
            if self.config.recorder.save_scores:
                self._save_scores(csid_pred, csid_conf, csid_gt, dataset_name)
            id_pred = np.concatenate([id_pred, csid_pred])
            id_conf = np.concatenate([id_conf, csid_conf])
            id_gt = np.concatenate([id_gt, csid_gt])

        # compute accuracy on csid
        print(u'\u2500' * 70, flush=True)
        self.eval_csid_acc(net, ood_data_loaders['csid'])

        # load nearood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='nearood')

        # load farood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='farood')
