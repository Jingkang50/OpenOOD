import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .metrics import auc, detection, fpr_recall


class Conf_Esti_Evaluator:
    def __init__(self, config: Config):
        self.config = config

    def report(self, test_metrics):
        print('''Complete testing,fpr95:{},auroc:{},
            aupr_in:{},aupr_out:{},detection_error:{}'''.format(
            test_metrics['fpr95'], test_metrics['auroc'],
            test_metrics['aupr_in'], test_metrics['aupr_out'],
            test_metrics['detection_error']))

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        net.eval()

        correct = []
        probability = []
        confidence = []
        eval_dataiter = iter(data_loader)
        with torch.no_grad():
            for step in tqdm(range(1,
                                   len(eval_dataiter) + 1),
                             desc='eval',
                             position=0,
                             leave=True):
                batch = next(eval_dataiter)
                images = Variable(batch['data']).cuda()
                labels = Variable(batch['label']).cuda()

                pred, conf = net(images)
                pred = F.softmax(pred, dim=-1)
                conf = torch.sigmoid(conf).data.view(-1)

                pred_value, pred = torch.max(pred.data, 1)
                correct.extend((pred == labels).cpu().numpy())
                probability.extend(pred_value.cpu().numpy())
                confidence.extend(conf.cpu().numpy())

        correct = np.array(correct).astype(bool)
        probability = np.array(probability)
        confidence = np.array(confidence)

        val_acc = np.mean(correct)
        metrics = {}
        metrics['acc'] = val_acc
        metrics['epoch_idx'] = epoch_idx
        return metrics

    def eval_ood(self, net, id_loader_dict, ood_loader_dict, epoch_idx=-1):

        id_loader = id_loader_dict['test']
        ood_loader = ood_loader_dict['val']
        ind_scores = []
        ood_scores = []
        net.eval()
        eval_dataiter_id = iter(id_loader)
        eval_dataiter_ood = iter(ood_loader)

        for step in tqdm(range(1,
                               len(eval_dataiter_id) + 1),
                         desc='id',
                         position=0,
                         leave=True):
            batch = next(eval_dataiter_id)
            images, labels = batch['data'], batch['label']

            images = Variable(images, requires_grad=True).cuda()
            images.retain_grad()
            if self.config['mode'] == 'baseline':
                pred, _ = net(images)
                pred = F.softmax(pred, dim=-1)
                pred = torch.max(pred.data, 1)[0]
                pred = pred.cpu().numpy()
                ind_scores.append(pred)
            else:
                _, confidence = net(images)
                confidence = torch.sigmoid(confidence)
                confidence = confidence.data.cpu().numpy()
                ind_scores.append(confidence)

        for step in tqdm(range(1,
                               len(eval_dataiter_ood) + 1),
                         desc='ood',
                         position=0,
                         leave=True):
            batch = next(eval_dataiter_ood)
            images, labels = batch['data'], batch['label']

            images = Variable(images, requires_grad=True).cuda()
            images.retain_grad()

            if self.config['mode'] == 'baseline':
                pred, _ = net(images)
                pred = F.softmax(pred, dim=-1)
                pred = torch.max(pred.data, 1)[0]
                pred = pred.cpu().numpy()
                ood_scores.append(pred)
            else:
                _, confidence = net(images)
                confidence = torch.sigmoid(confidence)
                confidence = confidence.data.cpu().numpy()
                ood_scores.append(confidence)

        ind_scores = np.concatenate(ind_scores)
        ood_scores = np.concatenate(ood_scores)

        ind_labels = np.ones(ind_scores.shape[0])
        ood_labels = np.ones(ood_scores.shape[0]) * (-1)

        labels = np.concatenate([ind_labels, ood_labels])
        scores = np.concatenate([ind_scores, ood_scores])

        fpr95, _ = fpr_recall(scores, labels, 0.95)
        auroc, aupr_in, aupr_out = auc(scores, labels)
        detection_error, best_delta = detection(ind_scores, ood_scores)

        metrics = {}
        metrics['fpr95'] = fpr95
        metrics['auroc'] = auroc
        metrics['aupr_in'] = aupr_in
        metrics['aupr_out'] = aupr_out
        metrics['detection_error'] = detection_error

        return metrics
