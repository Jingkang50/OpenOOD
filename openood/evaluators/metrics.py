import numpy as np
from sklearn import metrics

any_float = float | np.floating

def compute_all_metrics(conf, label, pred):
    np.set_printoptions(precision=3)
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, recall)

    accuracy = acc(pred, label)

    results = [fpr, auroc, aupr_in, aupr_out, accuracy]

    return results


def compute_der(*, conf: np.ndarray, label: np.ndarray, pred: np.ndarray, 
                p: float = 0.95, 
                id_pred: np.ndarray | None = None) -> np.floating:
    id_pred = conf[label != -1] if id_pred is None else id_pred
    gamma = np.quantile(id_pred, p)
    y_cor = label == pred
    der = detection_error_rate(y_cor= y_cor, ood_conf= conf, gamma = gamma)
    return der

def detection_error_rate(*, y_cor: np.ndarray, ood_conf: np.ndarray, gamma: np.floating) -> np.floating:
    """
    As described in the paper
    "Rethinking Out-of-Distribution Detection From a Human-Centric Perspective" 
    https://arxiv.org/pdf/2211.16778.pdf
    """

    tp_cor = np.sum(y_cor * (ood_conf >= gamma))
    fn_cor = np.sum(y_cor * (ood_conf < gamma))
    fp_cor = np.sum((1 - y_cor) * (ood_conf >= gamma))
    tn_cor = np.sum((1 - y_cor) * (ood_conf < gamma))


    DER = (fn_cor + fp_cor) / (tp_cor + fn_cor + fp_cor + tn_cor)

    return DER



# accuracy
def acc(pred, label) -> np.floating:
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc


# fpr_recall
def fpr_recall(conf, label, tpr):
    gt = np.ones_like(label)
    gt[label == -1] = 0

    fpr_list, tpr_list, threshold_list = metrics.roc_curve(gt, conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr)]
    thresh = threshold_list[np.argmax(tpr_list >= tpr)]
    return fpr, thresh


# auc
def auc_and_fpr_recall(conf, label, tpr_th) -> tuple[any_float, any_float, any_float, any_float]:
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    fpr : np.floating = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(1 - ood_indicator, conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(ood_indicator, -conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr


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
        tpr = np.sum(np.sum(X1 < delta)) / len(X1)
        error2 = np.sum(np.sum(Y1 > delta)) / len(Y1)
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
