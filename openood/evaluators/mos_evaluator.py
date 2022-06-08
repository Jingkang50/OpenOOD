from typing import Dict

import numpy as np
import sklearn.metrics as sk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .base_evaluator import BaseEvaluator


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

    smax = torch.nn.Softmax(dim=-1).cuda()
    for i in range(num_groups):
        group_logit = logits[:, group_slices[i][0]:group_slices[i][1]]

        group_softmax = smax(group_logit)
        group_others_score = group_softmax[:, 0]

        all_group_ood_score_MOS.append(-group_others_score)


    all_group_ood_score_MOS = torch.stack(all_group_ood_score_MOS, dim=1)
    final_max_score_MOS, _ = torch.max(all_group_ood_score_MOS, dim=1)
    return final_max_score_MOS.data.cpu().numpy()


def iterate_data(data_loader, model, group_slices):
    confs_mos = []
    train_dataiter = iter(data_loader)
    for train_step in tqdm(range(1,
                                 len(train_dataiter) + 1),
                           desc='Epoch {:03d}: '.format(0),
                           position=0,
                           leave=True):
        batch = next(train_dataiter)
        data = batch['data'].cuda()

        with torch.no_grad():

            logits = model(data)
            conf_mos = cal_ood_score(logits, group_slices)
            confs_mos.extend(conf_mos)

    return np.array(confs_mos)


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=1.):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl],
                                         1], np.r_[fps[sl],
                                                   0], np.r_[tps[sl],
                                                             0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true))
                          )  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(in_examples, out_examples):
    num_in = in_examples.shape[0]
    num_out = out_examples.shape[0]

    labels = np.zeros(num_in + num_out, dtype=np.int32)
    labels[:num_in] += 1

    examples = np.squeeze(np.vstack((in_examples, out_examples)))
    aupr_in = sk.average_precision_score(labels, examples)
    auroc = sk.roc_auc_score(labels, examples)

    recall_level = 0.95
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    labels_rev = np.zeros(num_in + num_out, dtype=np.int32)
    labels_rev[num_in:] += 1
    examples = np.squeeze(-np.vstack((in_examples, out_examples)))
    aupr_out = sk.average_precision_score(labels_rev, examples)
    return auroc, aupr_in, aupr_out, fpr


def run_eval(model, in_loader, out_loader, group_slices):
    # switch to evaluate mode
    model.eval()
    print('Running test...')
    print('Processing in-distribution data...')

    in_confs = iterate_data(in_loader, model, group_slices)

    print('Processing out-of-distribution data...')
    out_confs = iterate_data(out_loader, model, group_slices)

    in_examples = in_confs.reshape((-1, 1))
    out_examples = out_confs.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

    print('============Results for MOS============')
    print('AUROC: {}'.format(auroc))
    print('AUPR (In): {}'.format(aupr_in))
    print('AUPR (Out): {}'.format(aupr_out))
    print('FPR95: {}'.format(fpr95))


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
                           leave=True):
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

    def cal_group_slices(self, train_loader):
        config = self.config
         # if specified group_config
        if (config.trainer.group_config.endswith('npy')):
            classes_per_group = np.load(config.trainer.group_config)
        elif(config.trainer.group_config.endswith('txt')):
            classes_per_group = np.loadtxt(config.trainer.group_config, dtype=int)
        else:
            # cal group config
            config = self.config
            group = {}
            train_dataiter = iter(train_loader)
            for train_step in tqdm(range(1,
                                            len(train_dataiter) + 1),
                                    desc='cal group_config',
                                    position=0,
                                    leave=True):
                batch = next(train_dataiter)
                data = batch['data'].cuda()
                group_label = batch['group_label'].cuda()
                class_label = batch['class_label'].cuda()

                
                for i in range(len(class_label)):
                    try:
                        group[str(group_label[i].cpu().detach().numpy().tolist())]
                    except:
                        group[str(group_label[i].cpu().detach().numpy().tolist())] = []
                    
                    if class_label[i].cpu().detach().numpy().tolist() \
                            not in group[str(group_label[i].cpu().detach().numpy().tolist())]:
                        group[str(group_label[i].cpu().detach().numpy().tolist())].append(class_label[i].cpu().detach().numpy().tolist())

            classes_per_group=[]
            for i in range(len(group)):
                classes_per_group.append(max(group[str(i)])+1)

        self.num_groups = len(classes_per_group)
        self.group_slices = get_group_slices(classes_per_group)
        self.group_slices = self.group_slices.cuda()




    def eval_ood(self,
                 net: nn.Module,
                 id_data_loader: DataLoader,
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor=None):
        net = net.cuda()
        net.eval()
        self.cal_group_slices(id_data_loader['train'])
        dataset_name = self.config.dataset.name

        print(f'Performing inference on {dataset_name} dataset...', flush=True)

        run_eval(net, id_data_loader['val'], id_data_loader['test'],
                 self.group_slices)
        
        # test nearood
        for dataset_name, ood_dl in ood_data_loaders['nearood'].items():
            print(u'\u2500' * 70, flush=True)
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            run_eval(net, id_data_loader['val'], ood_dl,
                 self.group_slices)
        
        # test farood
        for dataset_name, ood_dl in ood_data_loaders['farood'].items():
            print(u'\u2500' * 70, flush=True)
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            run_eval(net, id_data_loader['val'], ood_dl,
                 self.group_slices)


    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        net.eval()
        self.cal_group_slices(data_loader)

        loss, top1 = run_eval_acc(net, data_loader, self.group_slices, self.num_groups)
        
        metrics = {}
        metrics['acc'] = np.mean(top1)
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = np.mean(loss)

        return metrics

    def report(self, test_metrics):
        print('Completed!', flush=True)
