from typing import Callable, List, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.evaluators.metrics import compute_all_metrics
from openood.postprocessors import BasePostprocessor

from .datasets import DATA_INFO, get_id_ood_dataloader
from .postprocessor import get_postprocessor
from .preprocessor import get_default_preprocessor, default_preprocessing_dict


class Evaluator:
    def __init__(
        self,
        net: nn.Module,
        id_name: str,
        data_root: str = './data',
        config_root: str = './configs',
        preprocessor: Callable = None,
        postprocessor_name: str = None,
        postprocessor: Type[BasePostprocessor] = None,
        batch_size: int = 200,
        shuffle: bool = False,
        num_workers: int = 4,
    ) -> None:
        # TODO
        """_summary_"""
        # check the arguments
        if postprocessor_name is None and postprocessor is None:
            raise ValueError('Please pass postprocessor_name or postprocessor')
        if postprocessor_name is not None and postprocessor is not None:
            print(
                'Postprocessor_name is ignored because postprocessor is passed'
            )
        if id_name not in DATA_INFO:
            raise ValueError(f'Dataset [{id_name}] is not supported')

        # get data preprocessor
        if preprocessor is None:
            preprocessor = get_default_preprocessor(id_name)

        # load data
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        dataloader_dict = get_id_ood_dataloader(id_name, data_root,
                                                preprocessor, **loader_kwargs)

        # get postprocessor
        if postprocessor is None:
            postprocessor = get_postprocessor(
                config_root, postprocessor_name,
                DATA_INFO[id_name]['num_classes'],
                default_preprocessing_dict[id_name]['normalization'][-1])
        if not isinstance(postprocessor, BasePostprocessor):
            raise TypeError(
                'postprocessor should inherit BasePostprocessor in OpenOOD')

        # postprocessor setup
        postprocessor.setup(net, dataloader_dict['id'], dataloader_dict['ood'])

        self.id_name = id_name
        self.net = net
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataloader_dict = dataloader_dict
        self.metrics = {
            'id_acc': None,
            'csid_acc': None,
            'id_ood': {
                'near': None,
                'far': None,
                'overall': None
            },
            'csid_ood': {
                'near': None,
                'far': None,
                'overall': None
            }
        }
        self.scores = {
            'id': {
                'train': None,
                'val': None,
                'test': None
            },
            'csid': {k: None
                     for k in dataloader_dict['csid'].keys()},
            'ood': {
                'val': None,
                'near':
                {k: None
                 for k in dataloader_dict['ood']['near'].keys()},
                'far': {k: None
                        for k in dataloader_dict['ood']['far'].keys()},
            },
            'id_preds': None,
            'id_labels': None,
            'csid_preds': {k: None
                           for k in dataloader_dict['csid'].keys()},
            'csid_labels': {k: None
                            for k in dataloader_dict['csid'].keys()},
        }
        self.hyperparam_search_flag = False
        self.net.eval()

        # how to ensure the postprocessors can work with
        # models whose definition doesn't align with OpenOOD

    def _classifier_inference(self,
                              data_loader: DataLoader,
                              msg: str = 'Acc Eval',
                              progress: bool = True):
        self.net.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=msg, disable=not progress):
                data = batch['data'].cuda()
                logits = self.net(data)
                preds = logits.argmax(1)
                all_preds.append(preds.cpu())
                all_labels.append(batch['label'])

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return all_preds, all_labels

    def eval_acc(self, csid: bool = False) -> float:
        if not csid:
            if self.metrics['id_acc'] is not None:
                return self.metrics['id_acc']
            else:
                if self.scores['id_preds'] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict['id']['test'], 'ID Acc Eval')
                    self.scores['id_preds'] = all_preds
                    self.scores['id_labels'] = all_labels
                else:
                    all_preds = self.scores['id_preds']
                    all_labels = self.scores['id_labels']

                assert len(all_preds) == len(all_labels)
                correct = (all_preds == all_labels).sum().item()
                acc = correct / len(all_labels) * 100
                self.metrics['id_acc'] = acc
                return acc
        else:
            if self.metrics['csid_acc'] is not None:
                return self.metrics['csid_acc']
            else:
                correct, total = 0
                for _, (dataname, dataloader) in enumerate(
                        self.dataloader_dict['csid'].items()):
                    if self.scores['csid_preds'][dataname] is None:
                        all_preds, all_labels = self._classifier_inference(
                            dataloader, f'CSID {dataname} Acc Eval')
                        self.scores['csid_preds'][dataname] = all_preds
                        self.scores['csid_labels'][dataname] = all_labels
                    else:
                        all_preds = self.scores['csid_preds'][dataname]
                        all_labels = self.scores['csid_labels'][dataname]

                    assert len(all_preds) == len(all_labels)
                    c = (all_preds == all_labels).sum().item()
                    t = len(all_labels)
                    correct += c
                    total += t

                acc = correct / total * 100
                self.metrics['csid_acc'] = acc
                return acc

    def eval_ood(self, csid: bool = False, progress: bool = True):
        id_name = 'id' if not csid else 'csid'
        if self.metrics[f'{id_name}_ood']['overall'] is None:
            self.net.eval()

            # perform hyperparameter search if have not done so
            if self.postprocessor.APS_mode and not self.hyperparam_search_flag:
                self.hyperparam_search()
                self.hyperparam_search_flag = True

            # id score
            if self.scores['id']['test'] is None:
                print(f'Performing inference on {self.id_name} test set...',
                      flush=True)
                id_pred, id_conf, id_gt = self.postprocessor.inference(
                    self.net, self.dataloader_dict['id']['test'], progress)
                self.scores['id']['test'] = [id_pred, id_conf, id_gt]
            else:
                id_pred, id_conf, id_gt = self.scores['id']['test']

            if csid:
                csid_pred, csid_conf, csid_gt = [], [], []
                for i, dataset_name in enumerate(self.scores['csid'].keys()):
                    if self.scores['csid'][dataset_name] is None:
                        print(
                            f'Performing inference on {self.id_name} (cs) test set {i+1}...',
                            flush=True)
                        temp_pred, temp_conf, temp_gt = self.postprocessor.inference(
                            self.net,
                            self.dataloader_dict['csid'][dataset_name],
                            progress)
                        self.scores['csid'][dataset_name] = [
                            temp_pred, temp_conf, temp_gt
                        ]

                    csid_pred.append(self.scores['csid'][dataset_name][0])
                    csid_conf.append(self.scores['csid'][dataset_name][1])
                    csid_gt.append(self.scores['csid'][dataset_name][2])

                csid_pred = np.concatenate(id_pred)
                csid_conf = np.concatenate(id_conf)
                csid_gt = np.concatenate(id_gt)

                id_pred = np.concatenate((id_pred, csid_pred))
                id_conf = np.concatenate((id_conf, csid_conf))
                id_gt = np.concatenate((id_gt, csid_gt))

            # load nearood data and compute ood metrics
            near_metrics = self._eval_ood([id_pred, id_conf, id_gt],
                                          ood_split='near',
                                          progress=progress)
            # load farood data and compute ood metrics
            far_metrics = self._eval_ood([id_pred, id_conf, id_gt],
                                         ood_split='far',
                                         progress=progress)

            if self.metrics[f'{id_name}_acc'] is None:
                self.eval_acc(csid)
            near_metrics[:, -1] = np.array([self.metrics[f'{id_name}_acc']] *
                                           len(near_metrics))
            far_metrics[:, -1] = np.array([self.metrics[f'{id_name}_acc']] *
                                          len(far_metrics))

            self.metrics[f'{id_name}_ood']['overall'] = pd.DataFrame(
                np.concatenate([near_metrics, far_metrics], axis=0),
                index=list(self.dataloader_dict['ood']['near'].keys()) \
                    + ['nearood'] + list(self.dataloader_dict['ood']['far'].keys()) \
                    + ['farood'],
                columns=['FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT',
                         'CCR_4', 'CCR_3', 'CCR_2', 'CCR_1', 'ACC'],
            )
        else:
            print('Evaluation has already been done!')

        with pd.option_context(
                'display.max_rows', None, 'display.max_columns', None,
                'display.float_format',
                '{:,.2f}'.format):  # more options can be specified also
            print(self.metrics[f'{id_name}_ood']['overall'])

        return self.metrics[f'{id_name}_ood']['overall']

    def _eval_ood(self,
                  id_list: List[np.ndarray],
                  ood_split: str = 'near',
                  progress: bool = True):
        print(f'Processing {ood_split} ood...', flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        for dataset_name, ood_dl in self.dataloader_dict['ood'][
                ood_split].items():
            if self.scores['ood'][ood_split][dataset_name] is None:
                print(f'Performing inference on {dataset_name} dataset...',
                      flush=True)
                ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                    self.net, ood_dl, progress)
                self.scores['ood'][ood_split][dataset_name] = [
                    ood_pred, ood_conf, ood_gt
                ]
            else:
                print(
                    f'Inference has been performed on {dataset_name} dataset...',
                    flush=True)
                [ood_pred, ood_conf,
                 ood_gt] = self.scores['ood'][ood_split][dataset_name]

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            print(f'Computing metrics on {dataset_name} dataset...')
            ood_metrics = compute_all_metrics(conf, label, pred)
            metrics_list.append(ood_metrics)
            self._print_metrics(ood_metrics)

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0, keepdims=True)
        self._print_metrics(list(metrics_mean[0]))
        return np.concatenate([metrics_list, metrics_mean], axis=0) * 100

    def _print_metrics(self, metrics):
        [fpr, auroc, aupr_in, aupr_out,
         ccr_4, ccr_3, ccr_2, ccr_1, _] \
         = metrics

        # print ood metric results
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
              flush=True)
        print('CCR: {:.2f}, {:.2f}, {:.2f}, {:.2f},'.format(
            ccr_4 * 100, ccr_3 * 100, ccr_2 * 100, ccr_1 * 100),
              end=' ',
              flush=True)
        print(u'\u2500' * 70, flush=True)
        print('', flush=True)

    def hyperparam_search(self):
        print('Starting automatic parameter search...')
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0

        for name in self.postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1

        for name in hyperparam_names:
            hyperparam_list.append(self.postprocessor.args_dict[name])

        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)

        final_index = None
        for i, hyperparam in enumerate(hyperparam_combination):
            self.postprocessor.set_hyperparam(hyperparam)

            id_pred, id_conf, id_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['id']['val'])
            ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['ood']['val'])

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            auroc = ood_metrics[1]

            print('Hyperparam: {}, auroc: {}'.format(hyperparam, auroc))
            if auroc > max_auroc:
                final_index = i
                max_auroc = auroc

        self.postprocessor.set_hyperparam(hyperparam_combination[final_index])
        print('Final hyperparam: {}'.format(
            self.postprocessor.get_hyperparam()))

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results
