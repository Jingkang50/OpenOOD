import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from openood.evaluators.draem_evaluator import get_auroc


class DCAEEvaluator:
    def __init__(self, config) -> None:
        self.config = config

    def report(self, test_metrics):
        print('Complete testing, roc_auc:{}'.format(test_metrics['roc_auc']))

    def eval_ood(self,
                 net,
                 id_loader_dict,
                 ood_loader_dict,
                 postprocessor=None,
                 epoch_idx=-1):
        target_class = self.config['normal_class']
        id_loader = id_loader_dict['val']
        ood_loader = ood_loader_dict['val']
        label_score_id = []
        label_score_ood = []
        net.eval()
        eval_dataiter_id = iter(id_loader)
        eval_dataiter_ood = iter(ood_loader)
        with torch.no_grad():
            for step in tqdm(range(1,
                                   len(eval_dataiter_id) + 1),
                             desc='id',
                             position=0,
                             leave=True):
                batch = next(eval_dataiter_id)
                inputs, labels = batch['data'].cuda(), batch['label']
                outputs = net(inputs)
                scores = torch.sum((outputs - inputs)**2,
                                   dim=tuple(range(1, outputs.dim())))
                label_score_id += list(
                    zip(labels.cpu().data.numpy().tolist(),
                        scores.cpu().data.numpy().tolist()))
            for step in tqdm(range(1,
                                   len(eval_dataiter_ood) + 1),
                             desc='ood',
                             position=0,
                             leave=True):
                batch = next(eval_dataiter_ood)
                inputs, labels = batch['data'].cuda(), batch['label']
                outputs = net(inputs)
                scores = torch.sum((outputs - inputs)**2,
                                   dim=tuple(range(1, outputs.dim())))
                label_score_ood += list(
                    zip(labels.cpu().data.numpy().tolist(),
                        scores.cpu().data.numpy().tolist()))

        labels, scores = zip(*(label_score_id + label_score_ood))
        labels = np.array(labels)
        indx1 = labels == target_class
        indx2 = labels != target_class
        labels[indx1] = 1
        labels[indx2] = 0
        scores = np.array(scores)
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
        metrics = {}
        roc_auc = auc(fpr, tpr)
        roc_auc = round(roc_auc, 4)
        metrics['epoch_idx'] = epoch_idx
        metrics['roc_auc'] = roc_auc
        return metrics


class DSVDDEvaluator:
    def __init__(self, config) -> None:
        self.config = config

    def report(self, test_metrics):
        print('Complete testing, roc_auc:{}'.format(test_metrics['roc_auc']))

    def eval_ood(self,
                 net,
                 id_loader_dict,
                 ood_loader_dict,
                 postprocessor=None,
                 epoch_idx=-1):
        auroc = get_auroc(net, id_loader_dict['test'], ood_loader_dict['val'],
                          postprocessor)
        metrics = {'epoch_idx': epoch_idx, 'roc_auc': auroc}
        return metrics

    def _eval_ood(self,
                  net,
                  hyperpara,
                  id_loader_dict,
                  ood_loader_dict,
                  postprocessor=None,
                  epoch_idx=-1):
        target_class = self.config['normal_class']
        id_loader = id_loader_dict['val']
        ood_loader = ood_loader_dict['val']
        label_score_id = []
        label_score_ood = []
        net.eval()
        eval_dataiter_id = iter(id_loader)
        eval_dataiter_ood = iter(ood_loader)
        with torch.no_grad():
            for step in tqdm(range(1,
                                   len(eval_dataiter_id) + 1),
                             desc='id',
                             position=0,
                             leave=True):
                batch = next(eval_dataiter_id)
                inputs, labels = batch['data'].cuda(), batch['label']
                outputs = net(inputs)
                dist = torch.sum((outputs - hyperpara['c'])**2,
                                 dim=tuple(range(1, outputs.dim())))
                if self.config.objective == 'soft-boundary':
                    scores = dist - hyperpara['R']**2
                else:
                    scores = dist
                label_score_id += list(
                    zip(labels.cpu().data.numpy().tolist(),
                        scores.cpu().data.numpy().tolist()))
            for step in tqdm(range(1,
                                   len(eval_dataiter_ood) + 1),
                             desc='ood',
                             position=0,
                             leave=True):
                batch = next(eval_dataiter_ood)
                inputs, labels = batch['data'].cuda(), batch['label']
                outputs = net(inputs)
                dist = torch.sum((outputs - hyperpara['c'])**2,
                                 dim=tuple(range(1, outputs.dim())))
                if self.config.objective == 'soft-boundary':
                    scores = dist - hyperpara['R']**2
                else:
                    scores = dist
                label_score_ood += list(
                    zip(labels.cpu().data.numpy().tolist(),
                        scores.cpu().data.numpy().tolist()))
        labels, scores = zip(*(label_score_id + label_score_ood))
        labels = np.array(labels)
        indx1 = labels == target_class
        indx2 = labels != target_class
        labels[indx1] = 1
        labels[indx2] = 0
        scores = np.array(scores)
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
        metrics = {}
        roc_auc = auc(fpr, tpr)
        roc_auc = round(roc_auc, 4)
        metrics['epoch_idx'] = epoch_idx
        metrics['roc_auc'] = roc_auc
        return metrics
