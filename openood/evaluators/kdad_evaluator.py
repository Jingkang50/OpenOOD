import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from torch.autograd import Variable
from tqdm import tqdm


class KdadDetectionEvaluator:
    """It is the kdad evaluator for Anomaly Detection."""
    def __init__(self, config):
        self.config = config

    def report(self, test_metrics):
        print('Complete testing, roc_auc:{}'.format(test_metrics['roc_auc']))

    def eval_ood(self, net, id_loader_dict, ood_loader_dict, epoch_idx=-1):
        config = self.config
        id_loader = id_loader_dict['train']
        ood_loader = ood_loader_dict['val']
        normal_class = config['normal_class']
        lamda = config['lamda']
        direction_only = config['direction_loss_only']
        model = net['model']
        vgg = net['vgg']

        target_class = normal_class
        similarity_loss = torch.nn.CosineSimilarity()
        label_score_id = []
        label_score_ood = []
        model.eval()
        eval_dataiter_id = iter(id_loader)
        eval_dataiter_ood = iter(ood_loader)

        # start evaluation
        for step in tqdm(range(1,
                               len(eval_dataiter_id) + 1),
                         desc='id',
                         position=0,
                         leave=True):
            batch = next(eval_dataiter_id)
            X, Y = batch['data'], batch['label']

            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)
            X = Variable(X).cuda()
            output_pred = model.forward(X)
            output_real = vgg(X)
            y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[
                9], output_pred[12]
            y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]

            if direction_only:
                loss_1 = 1 - similarity_loss(
                    y_pred_1.view(y_pred_1.shape[0], -1),
                    y_1.view(y_1.shape[0], -1))
                loss_2 = 1 - similarity_loss(
                    y_pred_2.view(y_pred_2.shape[0], -1),
                    y_2.view(y_2.shape[0], -1))
                loss_3 = 1 - similarity_loss(
                    y_pred_3.view(y_pred_3.shape[0], -1),
                    y_3.view(y_3.shape[0], -1))
                total_loss = loss_1 + loss_2 + loss_3
            else:
                abs_loss_1 = torch.mean((y_pred_1 - y_1)**2, dim=(1, 2, 3))
                loss_1 = 1 - similarity_loss(
                    y_pred_1.view(y_pred_1.shape[0], -1),
                    y_1.view(y_1.shape[0], -1))
                abs_loss_2 = torch.mean((y_pred_2 - y_2)**2, dim=(1, 2, 3))
                loss_2 = 1 - similarity_loss(
                    y_pred_2.view(y_pred_2.shape[0], -1),
                    y_2.view(y_2.shape[0], -1))
                abs_loss_3 = torch.mean((y_pred_3 - y_3)**2, dim=(1, 2, 3))
                loss_3 = 1 - similarity_loss(
                    y_pred_3.view(y_pred_3.shape[0], -1),
                    y_3.view(y_3.shape[0], -1))
                total_loss = loss_1 + loss_2 + loss_3 + lamda * (
                    abs_loss_1 + abs_loss_2 + abs_loss_3)

            label_score_id += list(
                zip(Y.cpu().data.numpy().tolist(),
                    total_loss.cpu().data.numpy().tolist()))
        for step in tqdm(range(1,
                               len(eval_dataiter_ood) + 1),
                         desc='ood',
                         position=0,
                         leave=True):
            batch = next(eval_dataiter_ood)
            X, Y = batch['data'], batch['label']

            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)
            X = Variable(X).cuda()
            output_pred = model.forward(X)
            output_real = vgg(X)
            y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[
                9], output_pred[12]
            y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]

            if direction_only:
                loss_1 = 1 - similarity_loss(
                    y_pred_1.view(y_pred_1.shape[0], -1),
                    y_1.view(y_1.shape[0], -1))
                loss_2 = 1 - similarity_loss(
                    y_pred_2.view(y_pred_2.shape[0], -1),
                    y_2.view(y_2.shape[0], -1))
                loss_3 = 1 - similarity_loss(
                    y_pred_3.view(y_pred_3.shape[0], -1),
                    y_3.view(y_3.shape[0], -1))
                total_loss = loss_1 + loss_2 + loss_3
            else:
                abs_loss_1 = torch.mean((y_pred_1 - y_1)**2, dim=(1, 2, 3))
                loss_1 = 1 - similarity_loss(
                    y_pred_1.view(y_pred_1.shape[0], -1),
                    y_1.view(y_1.shape[0], -1))
                abs_loss_2 = torch.mean((y_pred_2 - y_2)**2, dim=(1, 2, 3))
                loss_2 = 1 - similarity_loss(
                    y_pred_2.view(y_pred_2.shape[0], -1),
                    y_2.view(y_2.shape[0], -1))
                abs_loss_3 = torch.mean((y_pred_3 - y_3)**2, dim=(1, 2, 3))
                loss_3 = 1 - similarity_loss(
                    y_pred_3.view(y_pred_3.shape[0], -1),
                    y_3.view(y_3.shape[0], -1))
                total_loss = loss_1 + loss_2 + loss_3 + lamda * (
                    abs_loss_1 + abs_loss_2 + abs_loss_3)

            label_score_ood += list(
                zip(Y.cpu().data.numpy().tolist(),
                    total_loss.cpu().data.numpy().tolist()))

        # compute roc_auc
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
