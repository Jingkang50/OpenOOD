import csv
import os
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from torchvision import transforms

from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .base_evaluator import BaseEvaluator
from .metrics import compute_all_metrics


class PatchCoreEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        super(PatchCoreEvaluator, self).__init__(config)

    def eval_ood(self, net: nn.Module, id_data_loader: DataLoader,
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor: BasePostprocessor):
        net.eval()

        dataset_name = self.config.dataset.name
        print(f'Performing inference on {dataset_name} dataset...', flush=True)
        id_pred, id_conf, id_gt = postprocessor.inference(
            net, id_data_loader['patchTest'])  # not good
        good_pred, good_conf, good_gt = postprocessor.inference(
            net, id_data_loader['patchTestGood'])  # good

        pred = np.concatenate([id_pred, good_pred])
        conf = np.concatenate([id_conf, good_conf])
        gt = np.concatenate([id_gt, good_gt])

        self.gt_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.CenterCrop(224)
        ])
        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=mean_train, std=std_train)
        ])
        count = 0
        self.gt_list_px_lvl = []

        for batch in id_data_loader['patchGT']:
            #data = batch['data'].cuda()
            data = []
            label = batch['label'].cuda()
            name = batch['image_name']
            for i in name:
                path = os.path.join('./data/images/', i)
                gt_img = Image.open(path)
                gt_img = self.gt_transform(gt_img)
                gt_img = torch.unsqueeze(gt_img, 0)
                gt_np = gt_img.cpu().numpy()[0, 0].astype(int)
                count = count + 1
                self.gt_list_px_lvl.extend(gt_np.ravel())

        for i in good_gt:
            img = Image.open('./data/images/mvtec/hazelnut/test/good/000.png'
                             ).convert('RGB')
            img = self.transform(img)
            gt_img = torch.zeros([1, img.size()[-2], img.size()[-2]])
            gt_img = torch.unsqueeze(gt_img, 0)

            # gt_img = self.gt_transform(gt_img)
            gt_np = gt_img.cpu().numpy()[0, 0].astype(int)
            self.gt_list_px_lvl.extend(gt_np.ravel())

        self.pred_list_px_lvl = []
        self.pred_list_img_lvl = []

        for patchscore in conf:

            anomaly_map = patchscore[:, 0].reshape((28, 28))
            N_b = patchscore[np.argmax(patchscore[:, 0])]
            w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
            score = w * max(patchscore[:, 0])  # Image-level score

            anomaly_map_resized = cv2.resize(anomaly_map, (224, 224))
            anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized,
                                                       sigma=4)
            self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
            self.pred_list_img_lvl.append(score)

        print('Total image-level auc-roc score :')
        img_auc = roc_auc_score(gt, self.pred_list_img_lvl)
        print(img_auc)

        print('Total pixel-level auc-roc score :')
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        print(pixel_auc)

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        net.eval()
        id_pred, _, id_gt = postprocessor.inference(net, data_loader)
        metrics = {}
        metrics['acc'] = sum(id_pred == id_gt) / len(id_pred)
        metrics['epoch_idx'] = epoch_idx
        return metrics

    def report(self, test_metrics):
        print('Completed!', flush=True)
