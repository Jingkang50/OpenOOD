import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from openood.postprocessors import BasePostprocessor
from openood.utils import Config


class DRAEMEvaluator():
    def __init__(self, config: Config):
        self.config = config

    def eval(self,
             net,
             data_loader: DataLoader,
             postprocessor: BasePostprocessor = None,
             epoch_idx: int = -1):
        net['generative'].eval()
        net['discriminative'].eval()

        img_dim = 256

        obj_ap_pixel_list = []
        obj_auroc_pixel_list = []
        obj_ap_image_list = []
        obj_auroc_image_list = []

        with open(self.config.dataset.test.imglist_pth) as imgfile:
            dataset_length = len(imgfile.readlines())

        total_pixel_scores = np.zeros((img_dim * img_dim * dataset_length))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * dataset_length))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        display_images = torch.zeros((16, 3, 256, 256)).cuda()
        display_gt_images = torch.zeros((16, 3, 256, 256)).cuda()
        display_out_masks = torch.zeros((16, 1, 256, 256)).cuda()
        display_in_masks = torch.zeros((16, 1, 256, 256)).cuda()
        cnt_display = 0
        display_indices = np.random.randint(len(data_loader), size=(16, ))

        for i_batch, sample_batched in enumerate(data_loader):

            gray_batch = sample_batched['data']['image'].cuda()

            is_normal = sample_batched['data']['has_anomaly'].detach().numpy()[
                0, 0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched['data']['mask']
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose(
                (1, 2, 0))

            gray_rec = net['generative'](gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = net['discriminative'](joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            if i_batch in display_indices:
                t_mask = out_mask_sm[:, 1:, :, :]
                display_images[cnt_display] = gray_rec[0].cpu().detach()
                display_gt_images[cnt_display] = gray_batch[0].cpu().detach()
                display_out_masks[cnt_display] = t_mask[0].cpu().detach()
                display_in_masks[cnt_display] = true_mask[0].cpu().detach()

                cnt_display += 1

            out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()

            out_mask_averaged = torch.nn.functional.avg_pool2d(
                out_mask_sm[:, 1:, :, :], 21, stride=1,
                padding=21 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)

            anomaly_score_prediction.append(image_score)

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) *
                               img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) *
                                  img_dim * img_dim] = flat_true_mask
            mask_cnt += 1

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
        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)

        metrics = {
            'epoch_idx': epoch_idx,
            'image_auc': auroc,
            'image_ap': ap,
            'pixel_auc': auroc_pixel,
            'pixel_ap': ap_pixel
        }
        return metrics
