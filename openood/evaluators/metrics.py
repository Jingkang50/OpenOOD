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
    # ind_conf = conf[label != -1]
    # ood_conf = conf[label == -1]
    # num_ind = len(ind_conf)
    # num_ood = len(ood_conf)
    gt = np.ones_like(label)
    gt[label == -1] = 0
    # recall_num = int(np.floor(tpr * num_ind))
    # thresh = np.sort(ind_conf)[-recall_num]
    # num_fp = np.sum(ood_conf > thresh)
    # fpr = num_fp / num_ood

    fpr_list, tpr_list, threshold_list = metrics.roc_curve(gt, conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr)]
    thresh = threshold_list[np.argmax(tpr_list >= tpr)]
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


def detection(ind_confidences,
              ood_confidences,
              n_iter=100000,
              return_data=False):
    # calculate the minimum detection error
    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / n_iter

    best_error = 1.0
    best_delta = None
    all_thresholds = []
    all_errors = []
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        detection_error = (tpr + error2) / 2.0

        if return_data:
            all_thresholds.append(delta)
            all_errors.append(detection_error)

        if detection_error < best_error:
            best_error = np.minimum(best_error, detection_error)
            best_delta = delta

    if return_data:
        return best_error, best_delta, all_errors, all_thresholds
    else:
        return best_error, best_delta
