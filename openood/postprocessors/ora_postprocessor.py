from typing import Any

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor


class ORAPostprocessor(BasePostprocessor):
    """ORA: OOD detection with Relative Angles.

    For each test sample, ORA projects the feature onto each class's
    decision boundary using the linear classifier weights, then measures
    the angle between (i) the feature vector and (ii) its boundary
    projection, both centered at the mean of the per-class training
    feature means. The OOD score aggregates these per-class angles via
    `aggregation` (mean | max | min). `mean` is the default and is the
    recommended choice for CE-trained backbones; `max` is recommended
    for SupCon-trained backbones.
    """
    def __init__(self, config):
        super(ORAPostprocessor, self).__init__(config)
        self.APS_mode = False
        self.setup_flag = False
        self.mean_of_class_means = None
        self.num_classes = None

        self.args = self.config.postprocessor.postprocessor_args
        self.aggregation = getattr(self.args, 'aggregation', 'mean')
        if self.aggregation not in ('mean', 'max', 'min'):
            raise ValueError(
                f"aggregation must be one of 'mean', 'max', 'min'; "
                f'got {self.aggregation!r}')
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()
            num_classes = net.fc.weight.shape[0]

            class_sum = torch.zeros(num_classes, net.fc.weight.shape[1]).cuda()
            class_count = torch.zeros(num_classes).cuda()

            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda().float()
                    labels = batch['label'].cuda()
                    _, feature = net(data, return_feature=True)

                    class_sum.index_add_(0, labels, feature)
                    class_count.index_add_(
                        0, labels, torch.ones_like(labels, dtype=torch.float))

            class_means = class_sum / class_count.clamp(min=1).unsqueeze(1)
            self.mean_of_class_means = class_means.mean(dim=0)
            self.num_classes = num_classes
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feature = net(data, return_feature=True)
        preds = output.argmax(dim=1)
        max_logits = output.max(dim=1).values
        w = net.fc.weight  # (C, D)

        centered_feats = feature - self.mean_of_class_means
        norm_centered_feats = F.normalize(centered_feats, p=2, dim=1)

        trajectory = torch.zeros(feature.size(0),
                                 self.num_classes,
                                 device=feature.device)
        for c in range(self.num_classes):
            logit_diff = max_logits - output[:, c]
            weight_diff = w[preds] - w[c]
            weight_diff_norm_sq = (weight_diff * weight_diff).sum(dim=1)
            scale = (logit_diff / weight_diff_norm_sq).unsqueeze(1)
            feats_db = feature - scale * weight_diff

            centered_db = feats_db - self.mean_of_class_means
            norm_centered_db = F.normalize(centered_db, p=2, dim=1)
            cos_sim = (norm_centered_feats * norm_centered_db).sum(dim=1)
            trajectory[:,
                       c] = torch.arccos(cos_sim.clamp(-1.0, 1.0)) / torch.pi

        # Diagonal entries (c == preds) are 0/0 -> NaN. For mean/max we
        # treat them as 0 (matches the reference); for min we exclude them
        # since 0 would otherwise dominate.
        if self.aggregation == 'min':
            trajectory = torch.where(torch.isnan(trajectory),
                                     torch.full_like(trajectory, float('inf')),
                                     trajectory)
            score = trajectory.min(dim=1).values
        else:
            trajectory = torch.nan_to_num(trajectory, nan=0.0)
            if self.aggregation == 'max':
                score = trajectory.max(dim=1).values
            else:
                score = trajectory.mean(dim=1)
        return preds, score

    def set_hyperparam(self, hyperparam: list):
        self.aggregation = hyperparam[0]

    def get_hyperparam(self):
        return self.aggregation
