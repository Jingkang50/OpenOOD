import os

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


class ECEEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        """OOD Evaluator.

        Args:
            config (Config): Config file from
        """
        super(ECEEvaluator, self).__init__(config)

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1,
                 num_bins: int = 15):
        net.eval()
        """Calculates ECE.
        Args:
          num_bins: the number of bins to partition all samples. we set it as 15.
        Returns:
          ece: the calculated ECE value.
        """

        loss_avg = 0.0
        correct = 0
        total_scores = []
        total_preds = []
        total_labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True):
                # prepare data
                data = batch['data'].cuda()
                target = batch['label'].cuda()

                # forward
                output = net(data)
                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]
                score = output.data.max(1)[0]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

                total_preds.append(pred.cpu().numpy().reshape(-1))
                total_scores.append(score.cpu().numpy().reshape(-1))
                total_labels.append(target.data.cpu().numpy().reshape(-1))

        scores_np = np.reshape(total_scores, -1)
        preds_np = np.reshape(total_preds, -1)
        labels_np = np.reshape(total_labels, -1)
        acc_tab = np.zeros(num_bins)  # Empirical (true) confidence
        mean_conf = np.zeros(num_bins)  # Predicted confidence
        nb_items_bin = np.zeros(num_bins)  # Number of items in the bins
        tau_tab = np.linspace(0, 1, num_bins + 1)  # Confidence bins
        for i in np.arange(num_bins):  # Iterates over the bins
            # Selects the items where the predicted max probability falls in the bin
            # [tau_tab[i], tau_tab[i + 1)]
            sec = (tau_tab[i + 1] > scores_np) & (scores_np >= tau_tab[i])
            nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
            # Selects the predicted classes, and the true classes
            class_pred_sec, y_sec = preds_np[sec], labels_np[sec]
            # Averages of the predicted max probabilities
            mean_conf[i] = np.mean(
                scores_np[sec]) if nb_items_bin[i] > 0 else np.nan
            # Computes the empirical confidence
            acc_tab[i] = np.mean(
                class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan
        # Cleaning
        mean_conf = mean_conf[nb_items_bin > 0]
        acc_tab = acc_tab[nb_items_bin > 0]
        nb_items_bin = nb_items_bin[nb_items_bin > 0]
        if sum(nb_items_bin) != 0:
            ece = np.average(np.absolute(mean_conf - acc_tab),
                             weights=nb_items_bin.astype(np.float) /
                             np.sum(nb_items_bin))
        else:
            ece = 0.0

        loss = loss_avg / len(data_loader)
        acc = correct / len(data_loader.dataset)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss)
        metrics['acc'] = self.save_metrics(acc)
        metrics['ece'] = self.save_metrics(ece)
        return metrics
