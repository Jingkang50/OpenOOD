import csv
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from openood.postprocessors import BasePostprocessor

from .metrics import compute_all_metrics


class BaseEvaluator:
    def __init__(
        self,
        net: nn.Module,
    ):
        self.net = net

    def inference(self, data_loader: DataLoader,
                  postprocessor: BasePostprocessor):
        pred_list, conf_list, label_list = [], [], []

        for batch in data_loader:
            data = batch['data'].cuda()
            label = batch['label'].cuda()

            pred, conf = postprocessor(self.net, data)

            for idx in range(len(data)):
                pred_list.append(pred[idx].cpu().tolist())
                conf_list.append(conf[idx].cpu().tolist())
                label_list.append(label[idx].cpu().tolist())

        # convert values into numpy array
        pred_list = np.array(pred_list, dtype=int)
        conf_list = np.array(conf_list)
        label_list = np.array(label_list, dtype=int)

        return pred_list, conf_list, label_list

    def eval_classification(
        self,
        data_loader: DataLoader,
    ):
        self.net.eval()

        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for batch in data_loader:
                data = batch['data'].cuda()
                target = batch['label'].cuda()

                # forward
                output = self.net(data)
                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

        metrics = {}
        metrics['test_loss'] = loss_avg / len(data_loader)
        metrics['test_accuracy'] = correct / len(data_loader.dataset)
        return metrics

    def eval_csid_classification(
        self,
        csid_loaders: List[DataLoader],
        csv_path: str = None,
    ):
        self.net.eval()

        for i, csid_dl in enumerate(csid_loaders):
            csid_name = csid_dl.dataset.name
            print(f'Computing accuracy on {csid_name} dataset...')
            correct = 0
            with torch.no_grad():
                for batch in csid_dl:
                    data = batch['data'].cuda()
                    target = batch['label'].cuda()
                    # forward
                    output = self.net(data)
                    # accuracy
                    pred = output.data.max(1)[1]
                    correct += pred.eq(target.data).sum().item()
            acc = correct / len(csid_dl.dataset)
            print(f'CSID set {csid_name}: Accuracy {100 * acc:.2f}%',
                  flush=True)
            if csv_path:
                self._log_acc_results(acc, csv_path, dataset_name=csid_name)

    def eval_ood(
        self,
        id_data_loaders: List[DataLoader],
        ood_data_loaders: List[DataLoader],
        postprocessor: BasePostprocessor = None,
        method: str = 'each',
        csv_path: str = None,
    ):
        self.net.eval()

        if postprocessor is None:
            postprocessor = BasePostprocessor()

        if method == 'each':
            results_matrix = []
            id_pred, id_conf, id_gt = np.array([]), [], []
            for id_data_loader in id_data_loaders:
                id_name = id_data_loader.dataset.name

                print(f'Performing inference on {id_name} dataset...')
                id_sub_pred, id_sub_conf, id_sub_gt = self.inference(
                    id_data_loader, postprocessor)

                if csv_path:
                    save_dir = csv_path[:-4]
                    os.makedirs(save_dir, exist_ok=True)
                    np.savez(os.path.join(save_dir, id_name),
                             pred=id_sub_pred,
                             conf=id_sub_conf,
                             label=id_sub_gt)

                id_pred = np.concatenate([id_pred, id_sub_pred])
                id_conf = np.concatenate([id_conf, id_sub_conf])
                id_gt = np.concatenate([id_gt, id_sub_gt])

            for i, ood_dl in enumerate(ood_data_loaders):
                ood_name = ood_dl.dataset.name

                print(f'Performing inference on {ood_name} dataset...')
                ood_pred, ood_conf, ood_gt = self.inference(
                    ood_dl, postprocessor)

                if csv_path:
                    save_dir = csv_path[:-4]
                    os.makedirs(save_dir, exist_ok=True)
                    np.savez(os.path.join(save_dir, ood_name),
                             pred=ood_pred,
                             conf=ood_conf,
                             label=ood_gt)

                pred = np.concatenate([id_pred, ood_pred])
                conf = np.concatenate([id_conf, ood_conf])
                label = np.concatenate([id_gt, -1 * np.ones_like(ood_gt)])

                print(f'Computing metrics on {ood_name} dataset...')
                results = compute_all_metrics(conf, label, pred)
                if csv_path:
                    self._log_results(results, csv_path, dataset_name=ood_name)

                results_matrix.append(results)

            results_matrix = np.array(results_matrix)

            print(f'Computing mean metrics...')
            results_mean = np.mean(results_matrix, axis=0)
            if csv_path:
                self._log_results(results_mean, csv_path, dataset_name='mean')

    def _log_results(self, results, csv_path, dataset_name=None):
        [fpr, auroc, aupr_in, aupr_out, ccr_4, ccr_3, ccr_2, ccr_1,
         accuracy] = results

        write_content = {
            'dataset': dataset_name,
            'FPR@95': '{:.2f}'.format(100 * fpr),
            'AUROC': '{:.2f}'.format(100 * auroc),
            'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
            'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
            'CCR_4': '{:.2f}'.format(100 * ccr_4),
            'CCR_3': '{:.2f}'.format(100 * ccr_3),
            'CCR_2': '{:.2f}'.format(100 * ccr_2),
            'CCR_1': '{:.2f}'.format(100 * ccr_1),
            'ACC': '{:.2f}'.format(100 * accuracy),
        }
        fieldnames = list(write_content.keys())
        # print(write_content, flush=True)

        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def _log_acc_results(self, acc, csv_path, dataset_name=None):
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
        # print(write_content, flush=True)

        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)
