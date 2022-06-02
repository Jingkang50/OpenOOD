import numpy as np
import torch
from sklearn.metrics import (auc, average_precision_score, roc_auc_score,
                             roc_curve)

from openood.postprocessors import BasePostprocessor
from openood.utils import Config


class DRAEMEvaluator():
    def __init__(self, config: Config):
        self.config = config

    def eval_ood(self,
                 net,
                 id_data_loader,
                 ood_data_loaders,
                 postprocessor,
                 epoch_idx: int = -1):

        data_loaders = [id_data_loader['test'], ood_data_loaders['val']]

        with torch.no_grad():
            if self.config.use_gt:
                if not self.config.image_auroc_only:
                    return self._eval_ood_all(net, data_loaders, postprocessor,
                                              epoch_idx)
            return self._eval_ood_with_postprocessor(net, id_data_loader,
                                                     ood_data_loaders,
                                                     postprocessor, epoch_idx)

    def _eval_ood_with_postprocessor(self,
                                     net,
                                     id_data_loader,
                                     ood_data_loaders,
                                     postprocessor,
                                     epoch_idx: int = -1):
        auroc = get_auroc(net, id_data_loader['test'], ood_data_loaders['val'],
                          postprocessor)
        metrics = {
            'epoch_idx': epoch_idx,
            'image_auroc': auroc,
        }
        return metrics

    def _eval_ood(self, net, data_loaders, postprocessor, epoch_idx: int = -1):
        # ensure the networks in eval mode
        net['generative'].eval()
        net['discriminative'].eval()

        anomaly_score_gt = []
        anomaly_score_prediction = []

        # start evaltuating
        temp = 0
        for data_loader in data_loaders:
            for i_batch, sample_batched in enumerate(data_loader):
                # prepare data
                gray_batch = sample_batched['data'].cuda()
                # print(sample_batched['label'].numpy()[0])
                # if (float(sample_batched['label'].numpy()[0]) == -1 or ):
                #     anomaly_score_gt.append(float(1))
                # else:
                #     anomaly_score_gt.append(float(0))
                # anomaly_score_gt.append(
                #         float(sample_batched['label'].numpy()[0]))
                anomaly_score_gt.append(float(temp))

                # forward
                gray_rec = net['generative'](gray_batch)
                joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

                out_mask = net['discriminative'](joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                # calculate image level scores
                out_mask_averaged = torch.nn.functional.avg_pool2d(
                    out_mask_sm[:, 1:, :, :], 21, stride=1,
                    padding=21 // 2).cpu().detach().numpy()
                image_score = np.max(out_mask_averaged)

                anomaly_score_prediction.append(image_score)
            temp = temp + 1

        # calculate final scores
        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)

        metrics = {
            'epoch_idx': epoch_idx,
            'image_auroc': auroc,
        }
        print('image_auroc: ' + str(auroc))
        auroc1 = self.get_auroc(net, data_loaders[0], data_loaders[1],
                                postprocessor)
        print('auroc: ' + str(auroc1))
        return metrics

    def _eval_ood_all(self,
                      net,
                      data_loaders,
                      postprocessor: BasePostprocessor = None,
                      epoch_idx: int = -1):

        # ensure the networks in eval mode
        net['generative'].eval()
        net['discriminative'].eval()

        img_dim = 256

        with open(self.config.dataset.test.imglist_pth) as imgfile:
            dataset_length = len(imgfile.readlines())
        with open(self.config.ood_dataset.val.imglist_pth) as imgfile:
            dataset_length = dataset_length + len(imgfile.readlines())

        total_pixel_scores = np.zeros((img_dim * img_dim * dataset_length))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * dataset_length))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        # start evaltuating
        for data_loader in data_loaders:
            for i_batch, sample_batched in enumerate(data_loader):
                # prepare data
                gray_batch = sample_batched['data']['image'].cuda()
                anomaly_score_gt.append(
                    float(sample_batched['label'].numpy()[0]))
                true_mask = sample_batched['data']['mask']
                true_mask_cv = true_mask.detach().numpy()[
                    0, :, :, :].transpose((1, 2, 0))

                # forward
                gray_rec = net['generative'](gray_batch)
                joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

                out_mask = net['discriminative'](joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()

                # calculate image level scores
                out_mask_averaged = torch.nn.functional.avg_pool2d(
                    out_mask_sm[:, 1:, :, :], 21, stride=1,
                    padding=21 // 2).cpu().detach().numpy()
                image_score = np.max(out_mask_averaged)

                anomaly_score_prediction.append(image_score)

                # calculate pxiel level scores (localization)
                flat_true_mask = true_mask_cv.flatten()
                flat_out_mask = out_mask_cv.flatten()
                total_pixel_scores[mask_cnt * img_dim *
                                   img_dim:(mask_cnt + 1) * img_dim *
                                   img_dim] = flat_out_mask
                total_gt_pixel_scores[mask_cnt * img_dim *
                                      img_dim:(mask_cnt + 1) * img_dim *
                                      img_dim] = flat_true_mask
                mask_cnt += 1

        # calculate final scores
        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt,
                                     anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim *
                                                      mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores,
                                           total_pixel_scores)

        metrics = {
            'epoch_idx': epoch_idx,
            'image_auroc': auroc,
            'image_ap': ap,
            'pixel_auc': auroc_pixel,
            'pixel_ap': ap_pixel
        }
        return metrics

    def report(self, test_metrics):
        if self.config.image_auroc_only:
            print('Complete Evaluation:\n'
                  '{}\n'
                  '==============================\n'
                  'AUC Image: {:.2f} \n'
                  '=============================='.format(
                      self.config.dataset.name,
                      100.0 * test_metrics['image_auroc']),
                  flush=True)
            print('Completed!', flush=True)

        else:
            print('Complete Evaluation:\n'
                  '{}\n'
                  '==============================\n'
                  'AUC Image: {:.2f} \nAP Image: {:.2f} \n'
                  'AUC Pixel: {:.2f} \nAP Pixel: {:.2f} \n'
                  '=============================='.format(
                      self.config.dataset.name,
                      100.0 * test_metrics['image_auroc'],
                      100.0 * test_metrics['image_ap'],
                      100.0 * test_metrics['pixel_auc'],
                      100.0 * test_metrics['pixel_ap']),
                  flush=True)
            print('Completed!', flush=True)


def get_auroc(net, id_data_loader, ood_data_loader, postprocessor):
    id_pred, id_conf, id_gt = postprocessor.inference(net, id_data_loader)
    ood_pred, ood_conf, ood_gt = postprocessor.inference(net, ood_data_loader)
    ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood

    conf = np.concatenate([id_conf, ood_conf])
    label = np.concatenate([id_gt, ood_gt])

    ind_indicator = np.zeros_like(label)
    ind_indicator[label != -1] = 1

    fpr, tpr, thresholds = roc_curve(ind_indicator, conf)

    auroc = auc(fpr, tpr)

    return auroc
