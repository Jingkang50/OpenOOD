import csv
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .base_evaluator import BaseEvaluator
from .metrics import compute_all_metrics


def topk(output, target, ks=(1, )):
    """Returns one boolean vector for each k, whether the target is within the
    output's top-k."""
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].max(0)[0] for k in ks]


def get_group_slices(classes_per_group):
    group_slices = []
    start = 0
    for num_cls in classes_per_group:
        end = start + num_cls + 1
        group_slices.append([start, end])
        start = end
    return torch.LongTensor(group_slices)


def cal_ood_score(logits, group_slices):
    num_groups = group_slices.shape[0]

    all_group_ood_score_MOS = []
    for i in range(num_groups):
        group_logit = logits[:, group_slices[i][0]:group_slices[i][1]]

        group_softmax = F.softmax(group_logit, dim=-1)
        group_others_score = group_softmax[:, 0]

        all_group_ood_score_MOS.append(-group_others_score)

    all_group_ood_score_MOS = torch.stack(all_group_ood_score_MOS, dim=1)
    final_max_score_MOS, _ = torch.max(all_group_ood_score_MOS, dim=1)
    return final_max_score_MOS.data.cpu().numpy()


def iterate_data(data_loader, model, group_slices):
    confs_mos = []
    dataiter = iter(data_loader)

    with torch.no_grad():
        for _ in tqdm(range(1,
                            len(dataiter) + 1),
                      desc='Batches',
                      position=0,
                      leave=True,
                      disable=not comm.is_main_process()):
            batch = next(dataiter)
            data = batch['data'].cuda()

            logits = model(data)
            conf_mos = cal_ood_score(logits, group_slices)
            confs_mos.extend(conf_mos)

    return np.array(confs_mos)


def calc_group_softmax_acc(logits, labels, group_slices):
    num_groups = group_slices.shape[0]
    loss = 0
    num_samples = logits.shape[0]

    all_group_max_score, all_group_max_class = [], []

    smax = torch.nn.Softmax(dim=-1).cuda()
    cri = torch.nn.CrossEntropyLoss(reduction='none').cuda()

    for i in range(num_groups):
        group_logit = logits[:, group_slices[i][0]:group_slices[i][1]]
        group_label = labels[:, i]
        loss += cri(group_logit, group_label)

        group_softmax = smax(group_logit)
        group_softmax = group_softmax[:, 1:]  # disregard others category
        group_max_score, group_max_class = torch.max(group_softmax, dim=1)
        group_max_class += 1  # shift the class index by 1

        all_group_max_score.append(group_max_score)
        all_group_max_class.append(group_max_class)

    all_group_max_score = torch.stack(all_group_max_score, dim=1)
    all_group_max_class = torch.stack(all_group_max_class, dim=1)

    final_max_score, max_group = torch.max(all_group_max_score, dim=1)

    pred_cls_within_group = all_group_max_class[torch.arange(num_samples),
                                                max_group]

    gt_class, gt_group = torch.max(labels, dim=1)

    selected_groups = (max_group == gt_group)

    pred_acc = torch.zeros(logits.shape[0]).bool().cuda()

    pred_acc[selected_groups] = (
        pred_cls_within_group[selected_groups] == gt_class[selected_groups])

    return loss, pred_acc


def run_eval_acc(model, data_loader, group_slices, num_group):
    # switch to evaluate mode
    model.eval()

    print('Running validation...')

    all_c, all_top1 = [], []

    train_dataiter = iter(data_loader)
    for train_step in tqdm(range(1,
                                 len(train_dataiter) + 1),
                           desc='Test: ',
                           position=0,
                           leave=True,
                           disable=not comm.is_main_process()):
        batch = next(train_dataiter)
        data = batch['data'].cuda()
        group_label = batch['group_label'].cuda()
        class_label = batch['class_label'].cuda()
        labels = []
        for i in range(len(group_label)):
            label = torch.zeros(num_group, dtype=torch.int64)
            label[group_label[i]] = class_label[i] + 1
            labels.append(label.unsqueeze(0))
        labels = torch.cat(labels, dim=0).cuda()

        with torch.no_grad():
            # compute output, measure accuracy and record loss.
            logits = model(data)
            if group_slices is not None:
                c, top1 = calc_group_softmax_acc(logits, labels, group_slices)
            else:
                c = torch.nn.CrossEntropyLoss(reduction='none')(logits, labels)
                top1 = topk(logits, labels, ks=(1, ))[0]

            all_c.extend(c.cpu())  # Also ensures a sync point.
            all_top1.extend(top1.cpu())

    model.train()
    # all_c is val loss
    # all_top1 is val top1 acc
    return all_c, all_top1


class MOSEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        super(MOSEvaluator, self).__init__(config)
        self.config = config
        self.num_groups = None
        self.group_slices = None
        self.acc = None

    def cal_group_slices(self, train_loader):
        config = self.config
        # if specified group_config
        if (config.trainer.group_config.endswith('npy')):
            classes_per_group = np.load(config.trainer.group_config)
        elif (config.trainer.group_config.endswith('txt')):
            classes_per_group = np.loadtxt(config.trainer.group_config,
                                           dtype=int)
        else:
            # cal group config
            config = self.config
            group = {}
            train_dataiter = iter(train_loader)
            for train_step in tqdm(range(1,
                                         len(train_dataiter) + 1),
                                   desc='cal group_config',
                                   position=0,
                                   leave=True,
                                   disable=not comm.is_main_process()):
                batch = next(train_dataiter)
                group_label = batch['group_label']
                class_label = batch['class_label']

                for i in range(len(class_label)):
                    gl = group_label[i].item()
                    cl = class_label[i].item()

                    try:
                        group[str(gl)]
                    except:
                        group[str(gl)] = []

                    if cl not in group[str(gl)]:
                        group[str(gl)].append(cl)

            classes_per_group = []
            for i in range(len(group)):
                classes_per_group.append(max(group[str(i)]) + 1)

        self.num_groups = len(classes_per_group)
        self.group_slices = get_group_slices(classes_per_group)
        self.group_slices = self.group_slices.cuda()

    def eval_ood(self,
                 net: nn.Module,
                 id_data_loader: DataLoader,
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor=None,
                 fsood=False):
        net.eval()
        if self.group_slices is None or self.num_groups is None:
            self.cal_group_slices(id_data_loader['train'])
        dataset_name = self.config.dataset.name

        print(f'Performing inference on {dataset_name} dataset...', flush=True)
        id_conf = iterate_data(id_data_loader['test'], net, self.group_slices)
        # dummy pred and gt
        # the accuracy will be handled by self.eval_acc
        id_pred = np.zeros_like(id_conf)
        id_gt = np.zeros_like(id_conf)

        if fsood:
            # load csid data and compute confidence
            for dataset_name, csid_dl in ood_data_loaders['csid'].items():
                print(f'Performing inference on {dataset_name} dataset...',
                      flush=True)
                csid_conf = iterate_data(csid_dl, net, self.group_slices)
                # dummy pred and gt
                # the accuracy will be handled by self.eval_acc
                csid_pred = np.zeros_like(csid_conf)
                csid_gt = np.zeros_like(csid_conf)
                if self.config.recorder.save_scores:
                    self._save_scores(csid_pred, csid_conf, csid_gt,
                                      dataset_name)
                id_pred = np.concatenate([id_pred, csid_pred])
                id_conf = np.concatenate([id_conf, csid_conf])
                id_gt = np.concatenate([id_gt, csid_gt])

        # load nearood data and compute ood metrics
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       ood_split='nearood')
        # load farood data and compute ood metrics
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       ood_split='farood')

    def _eval_ood(self,
                  net: nn.Module,
                  id_list: List[np.ndarray],
                  ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                  ood_split: str = 'nearood'):
        print(f'Processing {ood_split}...', flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        for dataset_name, ood_dl in ood_data_loaders[ood_split].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            ood_conf = iterate_data(ood_dl, net, self.group_slices)
            ood_gt = -1 * np.ones_like(ood_conf)  # hard set to -1 as ood
            # dummy pred
            ood_pred = np.zeros_like(ood_conf)
            if self.config.recorder.save_scores:
                self._save_scores(ood_pred, ood_conf, ood_gt, dataset_name)

            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            print(f'Computing metrics on {dataset_name} dataset...')

            ood_metrics = compute_all_metrics(conf, label, pred)
            # the acc here is not reliable
            # since we use dummy pred and gt for id samples
            # so we use the acc computed by self.eval_acc
            ood_metrics[-1] = self.acc

            if self.config.recorder.save_csv:
                self._save_csv(ood_metrics, dataset_name=dataset_name)
            metrics_list.append(ood_metrics)

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        if self.config.recorder.save_csv:
            self._save_csv(metrics_mean, dataset_name=ood_split)

    def _save_csv(self, metrics, dataset_name):
        [fpr, auroc, aupr_in, aupr_out, accuracy] = metrics

        write_content = {
            'dataset': dataset_name,
            'FPR@95': '{:.2f}'.format(100 * fpr),
            'AUROC': '{:.2f}'.format(100 * auroc),
            'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
            'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
            'ACC': '{:.2f}'.format(100 * accuracy)
        }

        fieldnames = list(write_content.keys())

        # print ood metric results
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
              flush=True)
        print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
        print(u'\u2500' * 70, flush=True)

        csv_path = os.path.join(self.config.output_dir, 'ood.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def _save_scores(self, pred, conf, gt, save_name):
        save_dir = os.path.join(self.config.output_dir, 'scores')
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, save_name),
                 pred=pred,
                 conf=conf,
                 label=gt)

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1,
                 num_groups: int = None,
                 group_slices: torch.Tensor = None,
                 fsood: bool = False,
                 csid_data_loaders: DataLoader = None):
        net.eval()
        if num_groups is None or group_slices is None:
            self.cal_group_slices(data_loader)
        else:
            self.num_groups = num_groups
            self.group_slices = group_slices.cuda()

        loss, top1 = run_eval_acc(net, data_loader, self.group_slices,
                                  self.num_groups)

        if fsood:
            assert csid_data_loaders is not None
            for dataset_name, csid_dl in csid_data_loaders.items():
                _, temp = run_eval_acc(net, csid_dl, self.group_slices,
                                       self.num_groups)
                top1.extend(temp)

        metrics = {}
        metrics['acc'] = np.mean(top1)
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = np.mean(loss)
        self.acc = metrics['acc']

        return metrics

    def report(self, test_metrics):
        print('Completed!', flush=True)
