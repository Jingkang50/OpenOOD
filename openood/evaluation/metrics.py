import csv
import os

import numpy as np
from sklearn import metrics


# accuracy
def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc


# fpr_recall
def fpr_recall(conf, label, tpr):
    ind_conf = conf[label != -1]
    ood_conf = conf[label == -1]
    num_ind = len(ind_conf)
    num_ood = len(ood_conf)

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf > thresh)
    fpr = num_fp / num_ood
    return fpr, thresh


def auc(conf, label):
    ind_indicator = np.zeros_like(label)
    ind_indicator[label != -1] = 1

    fpr, tpr, thresholds = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, thresholds_in = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, thresholds_out = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out


# ccr_fpr
def ccr_fpr(conf, fpr, pred, label):
    ind_conf = conf[label != -1]
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    ood_conf = conf[label == -1]

    num_ind = len(ind_conf)
    num_ood = len(ood_conf)

    fp_num = int(np.ceil(fpr * num_ood))
    thresh = np.sort(ood_conf)[-fp_num]
    num_tp = np.sum((ind_conf > thresh) * (ind_pred == ind_label))
    ccr = num_tp / num_ind

    return ccr


def compute_all_metrics(conf, label, pred, file_path=None, verbose=True):
    recall = 0.95
    fpr, thresh = fpr_recall(conf, label, recall)
    auroc, aupr_in, aupr_out = auc(conf, label)

    ccr_1 = ccr_fpr(conf, 0.1, pred, label)
    ccr_2 = ccr_fpr(conf, 0.01, pred, label)
    ccr_3 = ccr_fpr(conf, 0.001, pred, label)
    ccr_4 = ccr_fpr(conf, 0.0001, pred, label)

    accuracy = acc(pred, label)

    if verbose:
        print(
            'FPR@{}: {:.2f}, AUROC: {:.2f}, AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.
            format(recall, 100 * fpr, 100 * auroc, 100 * aupr_in,
                   100 * aupr_out))
        print('CCR: {:.2f}, {:.2f}, {:.2f}, {:.2f}, ACC: {:.2f}'.format(
            ccr_4 * 100, ccr_3 * 100, ccr_2 * 100, ccr_1 * 100,
            accuracy * 100))

    results = [
        fpr, auroc, aupr_in, aupr_out, ccr_4, ccr_3, ccr_2, ccr_1, accuracy
    ]

    if file_path:
        save_csv(file_path, results)

    return results


def save_csv(file_path, results):
    save_path = os.path.join(file_path, '..')
    filename = file_path.split('/')[-1]
    [fpr, auroc, aupr_in, aupr_out, ccr_4, ccr_3, ccr_2, ccr_1,
     accuracy] = results

    save_exp_name = os.path.join(
        save_path, 'summary_{}.csv'.format(file_path.split('/')[-2]))
    fieldnames = [
        'Experiment_PATH',
        'FPR@95',
        'AUROC',
        'AUPR_IN',
        'AUPR_OUT',
        'CCR@e4',
        'CCR@e3',
        'CCR@e2',
        'CCR@e1',
        'ACC',
    ]
    write_content = {
        'Experiment_PATH': filename,
        'FPR@95': round(100 * fpr, 2),
        'AUROC': round(100 * auroc, 2),
        'AUPR_IN': round(100 * aupr_in, 2),
        'AUPR_OUT': round(100 * aupr_out, 2),
        'CCR@e4': round(ccr_4 * 100, 2),
        'CCR@e3': round(ccr_3 * 100, 2),
        'CCR@e2': round(ccr_2 * 100, 2),
        'CCR@e1': round(ccr_1 * 100, 2),
        'ACC': round(accuracy * 100, 2),
    }

    if not os.path.exists(save_exp_name):
        with open(save_exp_name, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(write_content)
    else:
        with open(save_exp_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(write_content)
