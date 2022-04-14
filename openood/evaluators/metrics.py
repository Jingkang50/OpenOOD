import numpy as np
from sklearn import metrics


def compute_all_metrics(conf, label, pred):
    recall = 0.95
    fpr, thresh = fpr_recall(conf, label, recall)
    auroc, aupr_in, aupr_out = auc(conf, label)

    ccr_1 = ccr_fpr(conf, 0.1, pred, label)
    ccr_2 = ccr_fpr(conf, 0.01, pred, label)
    ccr_3 = ccr_fpr(conf, 0.001, pred, label)
    ccr_4 = ccr_fpr(conf, 0.0001, pred, label)

    accuracy = acc(pred, label)

    results = [
        fpr, auroc, aupr_in, aupr_out, ccr_4, ccr_3, ccr_2, ccr_1, accuracy
    ]

    return results


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


# auc
def auc(conf, label):

    ind_indicator = np.zeros_like(label)
    ind_indicator[label != -1] = 1
    
    fpr, tpr, thresholds = metrics.roc_curve(ind_indicator, conf)

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(ind_indicator, conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(1 - ind_indicator, 1 - conf)

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
