import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Any, Optional

from openood.postprocessors.base_postprocessor import BasePostprocessor


class CFOODPostprocessor(BasePostprocessor):
    """
    CFOOD-based post-hoc OOD detector.

    - setup():
        builds an ID reference set of features and precomputes k-NN distances
        of reference samples against themselves.
    - postprocess():
        computes a CFOOD score for each test sample and returns:
            pred: predicted ID class (argmax of softmax)
            conf: confidence score, defined as -CFOOD
                  (higher = more likely in-distribution).
    """

    def __init__(self, config):
        self.config = config
        self.postprocessor_args = getattr(
            config.postprocessor, "postprocessor_args", None
        )

        # default hyperparameters
        alpha = 0.1
        k = 0.01
        avg_topk = False

        if self.postprocessor_args is not None:
            alpha = float(getattr(self.postprocessor_args, "alpha", alpha))
            k = float(getattr(self.postprocessor_args, "k", k))
            avg_topk = bool(
                getattr(self.postprocessor_args, "avg_topk", avg_topk)
            )

        assert 0 < alpha <= 1.0, "alpha must be in (0, 1]."

        self.alpha = alpha
        self.k = k
        self.avg_topk = avg_topk  # reserved for future use

        self.ref: Optional[torch.Tensor] = None
        self.index_id: Optional[torch.Tensor] = None
        self.kval: Optional[np.ndarray] = None
        self.kpos: Optional[np.ndarray] = None
        self.klen: Optional[int] = None
        self.knndst: Optional[torch.Tensor] = None

        self.setup_flag = False
        self.has_data_based_setup = True

    @torch.no_grad()
    def setup(
        self,
        net: nn.Module,
        id_loader_dict,
        ood_loader_dict=None,
        id_loader_split: str = "train",
    ):
    
        if self.setup_flag:
            return

        net.eval()
        device = next(net.parameters()).device

        feats = []

        loader = id_loader_dict.get(id_loader_split, None)
        if loader is None:
            alt_split = "val" if id_loader_split == "train" else "train"
            loader = id_loader_dict.get(alt_split, None)

        if loader is None:
            raise RuntimeError(
                "CFOODPostprocessor.setup: cannot find an ID loader "
                f"('{id_loader_split}' or 'val')."
            )

        # extract ID features
        for batch in tqdm(loader, desc="CFOOD setup (extract ID feats)"):
            x = batch["data"].to(device, non_blocking=True).float()
            _, f = net(x, return_feature=True)
            feats.append(f.detach().cpu())

        feats = torch.cat(feats, dim=0)  # (N, D)

        # L2 normalization
        feats = feats / (torch.norm(feats, dim=1, keepdim=True) + 1e-10)

        N = feats.size(0)
        n = int(self.alpha * N) if self.alpha < 1.0 else N

        # optionally subsample the reference set
        if self.alpha < 1.0:
            perm = torch.randperm(N)[:n]
            self.ref = feats[perm].contiguous()
            self.index_id = perm
        else:
            self.ref = feats.contiguous()
            self.index_id = torch.arange(N)

        # grid of k-fractions for CFOOD
        kval = np.concatenate(
            (
                np.linspace(0.001, 0.1, 100),
                np.linspace(0.105, 0.5, 81),
                np.linspace(0.51, 0.99, 49),
            )
        )
        kpos = np.array(kval * n - 1, dtype=int)
        # remove duplicates / non-increasing positions
        keep = np.concatenate(([True], kpos[:-1] < kpos[1:]))
        self.kval = kval[keep]
        self.kpos = kpos[keep]
        self.klen = len(self.kpos)

        # precompute k-NN distances
        self.knndst = torch.zeros((n, self.klen), dtype=torch.float32)
        ref = self.ref  # (n, D)

        for i in tqdm(range(n), desc="CFOOD setup (self kNN)"):
            dst = torch.sum((ref[i] - ref) ** 2, dim=1)  # (n,)
            dst_sorted = torch.sort(dst, dim=0).values
            # +1 to skip self (distance 0)
            self.knndst[i, :] = dst_sorted[self.kpos + 1]

        self.setup_flag = True

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """
        Compute CFOOD scores for a batch of data.

        Returns:
            pred: LongTensor (B,) with ID predictions.
            conf: FloatTensor (B,) with confidence scores (higher = more ID).
        """
        device = next(net.parameters()).device

        x = data.to(device, non_blocking=True).float()
        logits, feat = net(x, return_feature=True)  # feat: (B, D)

        feat = feat.detach()
        feat = feat / (torch.norm(feat, dim=1, keepdim=True) + 1e-10)

        ref = self.ref.to(device)        # (n, D)
        knndst = self.knndst.to(device)  # (n, klen)
        n = ref.size(0)

        kval = torch.from_numpy(self.kval).to(device)  # (klen,)
        klen = self.klen

        scores = []
        for v in feat:  # v: (D,)
            dst = torch.sum((v - ref) ** 2, dim=1)  # (n,)

            # fraction of reference points whose k-NN distance >= distance of v
            count = (
                knndst >= dst.view(n, 1)
            ).sum(dim=0).float() / float(n)  # (klen,)

            pos = torch.nonzero(count >= self.k, as_tuple=False)
            if pos.numel() > 0:
                kappa = kval[pos[0, 0]]
            else:
                kappa = torch.tensor(1.0, device=device)

            scores.append(kappa)

        scores = torch.stack(scores, dim=0)  # (B,)
        conf = -scores

        # standard ID prediction
        pred = torch.softmax(logits, dim=1).argmax(dim=1)

        return pred, conf

    def set_hyperparam(self, hyperparam: list):
        if len(hyperparam) > 0:
            self.alpha = float(hyperparam[0])
        if len(hyperparam) > 1:
            self.k = float(hyperparam[1])

    def get_hyperparam(self):
        return [self.alpha, self.k]

