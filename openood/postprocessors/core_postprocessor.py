from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


class COREPostprocessor(BasePostprocessor):
    """CORE: Confidence + Orthogonal Residual Evidence.

    Decomposes penultimate features z relative to classifier weight w_c:
        z_parallel = (z . w_hat_c) * w_hat_c
        z_perp     = z - z_parallel

    Score = 0.5 * zscore(energy) + 0.5 * zscore(cos(z_perp, mu_perp[y_hat]))
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False
        self.eps = 1e-8

    def setup(self, net, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        net.eval()
        # Step 1: get classifier weights
        w, b = net.get_fc()
        W = torch.from_numpy(w).float().cuda()  # (C, D)
        self.w_numpy = w  # keep for logit reconstruction
        self.b_numpy = b
        self.w_normalized = F.normalize(W, p=2, dim=1)  # (C, D)

        # Step 2: collect training features and labels
        print('\n [CORE] Collecting training features...')
        all_feats, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(id_loader_dict['train'],
                              desc='CORE setup', position=0, leave=True):
                data = batch['data'].cuda()
                label = batch['label']
                logits, feat = net(data, return_feature=True)
                all_feats.append(feat.cpu())
                all_labels.append(label)
        all_feats = torch.cat(all_feats)    # (N, D)
        all_labels = torch.cat(all_labels)  # (N,)

        # Step 3: compute per-class mu_perp (on correctly classified samples)
        print(' [CORE] Computing per-class residual directions...')
        D = all_feats.shape[1]
        mu_perp = torch.zeros(self.num_classes, D)

        # Predict on training set to get correct mask
        logits_train = torch.from_numpy(
            all_feats.numpy() @ self.w_numpy.T + self.b_numpy).float()
        pred_train = logits_train.argmax(dim=1)
        correct_mask = pred_train == all_labels

        for c in range(self.num_classes):
            class_mask = (all_labels == c) & correct_mask
            if class_mask.sum() == 0:
                continue
            feats_c = all_feats[class_mask].cuda()  # (Nc, D)
            w_hat_c = self.w_normalized[c]           # (D,)
            proj = (feats_c @ w_hat_c).unsqueeze(1) * w_hat_c.unsqueeze(0)
            z_perp_c = feats_c - proj
            mu_perp[c] = z_perp_c.mean(dim=0).cpu()

        self.mu_perp = F.normalize(mu_perp, p=2, dim=1).cuda()  # (C, D)

        # Step 4: compute z-score stats on correct training samples
        print(' [CORE] Computing normalization statistics...')
        correct_feats = all_feats[correct_mask]
        correct_logits = logits_train[correct_mask]
        correct_preds = pred_train[correct_mask]

        I_all, T_all = [], []
        BATCH = 10000
        n = correct_feats.shape[0]
        for i in range(0, n, BATCH):
            feats_b = correct_feats[i:i + BATCH].cuda()
            logits_b = correct_logits[i:i + BATCH].cuda()
            preds_b = correct_preds[i:i + BATCH]

            # Energy
            I = torch.logsumexp(logits_b, dim=1)

            # Residual direction consistency
            w_hat = self.w_normalized[preds_b]     # (B, D)
            proj = (feats_b * w_hat).sum(dim=1, keepdim=True) * w_hat
            z_perp = feats_b - proj
            z_perp_norm = z_perp.norm(dim=1, keepdim=True)
            z_perp_hat = z_perp / (z_perp_norm + self.eps)
            mu = self.mu_perp[preds_b]             # (B, D)
            T = (z_perp_hat * mu).sum(dim=1)

            # Handle degenerate case
            degenerate = (z_perp_norm.squeeze() < self.eps)
            T[degenerate] = 1.0

            I_all.append(I.cpu())
            T_all.append(T.cpu())

        I_all = torch.cat(I_all)
        T_all = torch.cat(T_all)

        self.mean_I = I_all.mean().cuda()
        self.std_I = I_all.std().cuda()
        self.mean_T = T_all.mean().cuda()
        self.std_T = T_all.std().cuda()

        self.setup_flag = True
        print(' [CORE] Setup complete.')

    @torch.no_grad()
    def postprocess(self, net, data):
        logits, features = net(data, return_feature=True)
        pred = logits.argmax(dim=1)

        # Confidence: energy
        I = torch.logsumexp(logits, dim=1)

        # Membership: residual direction consistency
        w_hat = self.w_normalized[pred]  # (B, D)
        proj = (features * w_hat).sum(dim=1, keepdim=True) * w_hat
        z_perp = features - proj
        z_perp_norm = z_perp.norm(dim=1, keepdim=True)
        z_perp_hat = z_perp / (z_perp_norm + self.eps)
        mu = self.mu_perp[pred]          # (B, D)
        T = (z_perp_hat * mu).sum(dim=1)

        degenerate = (z_perp_norm.squeeze() < self.eps)
        T[degenerate] = 1.0

        # Z-score normalize and combine
        I_prime = (I - self.mean_I) / (self.std_I + self.eps)
        T_prime = (T - self.mean_T) / (self.std_T + self.eps)
        score = 0.5 * I_prime + 0.5 * T_prime

        return pred, score
