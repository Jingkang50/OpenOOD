import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class FortePostprocessor(BasePostprocessor):
    """Forte: Finding Outliers with Representation Typicality Estimation
    (Ganguly et al., ICLR 2025, https://openreview.net/forum?id=7XNgVPxCiA).

    Forte is a two-sample test: each evaluated set of samples is compared, as
    a sample, against a reference sample of ID data. Per-point PRDC summary
    statistics (precision, recall, density, coverage) are computed for every
    test point against the reference manifold, and a density model fitted on
    the PRDC statistics of held-out ID data scores typicality (higher = ID).

    Because per-point recall and coverage depend on each test point's k-NN
    radius *within the evaluated set*, scores are functions of the whole
    dataloader rather than of single samples. `inference` is therefore
    overridden to treat every evaluated dataloader as one sample of the
    two-sample test. The per-batch `postprocess` fallback applies the same
    computation at batch granularity and is only approximate for small
    batches.

    `setup` splits the ID train sample in half: one half is the reference
    manifold, the PRDC statistics of the other half against the reference
    form the fit set of the density model (KDE / GMM / OCSVM).

    Works with any feature space: a `ForteNet`-style module exposing
    `get_features` provides multiple representation spaces (PRDC statistics
    are concatenated across spaces, as in the paper); any standard OpenOOD
    classifier supporting `net(x, return_feature=True)` provides one.
    """
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.nearest_k = self.args.nearest_k
        self.variant = self.args.variant
        self.n_setup_samples = self.args.n_setup_samples
        self.density_fit_cap = self.args.density_fit_cap
        self.gmm_components = self.args.gmm_components
        self.seed = self.args.seed
        self.kde_jitter = getattr(self.args, 'kde_jitter', None) or 1e-3
        # optional directory for caching extracted features across runs
        self.feature_cache_dir = getattr(self.args, 'feature_cache_dir', None)
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.APS_mode = self.config.postprocessor.APS_mode
        self.setup_flag = False
        self.ref_feats = None
        self.ref_radii = None
        self.fit_prdc = None
        self.density_model = None
        # rows of pairwise-distance work processed at a time
        self.chunk_size = 4096

    # ------------------------------------------------------------------
    # feature extraction
    # ------------------------------------------------------------------
    def _rep_features(self, net: nn.Module, data: torch.Tensor):
        """List of (B, D) feature tensors, one per representation space."""
        if hasattr(net, 'get_features'):
            return [f.float() for f in net.get_features(data).values()]
        _, feature = net(data, return_feature=True)
        return [feature.float()]

    def _cache_key(self, data_loader: DataLoader, subsample_cap):
        dataset = data_loader.dataset
        name = getattr(dataset, 'name', type(dataset).__name__)
        subsampled = subsample_cap is not None and subsample_cap > 0 \
            and len(dataset) > subsample_cap
        # the seed only influences which samples are drawn when subsampling
        suffix = f'cap{subsample_cap}_seed{self.seed}' if subsampled else 'all'
        return f'{name}_{len(dataset)}_{suffix}'

    @torch.no_grad()
    def _extract_loader(self,
                        net: nn.Module,
                        data_loader: DataLoader,
                        subsample_cap: int = None,
                        progress: bool = True,
                        desc: str = 'Feature extraction'):
        cache_file = None
        if self.feature_cache_dir:
            os.makedirs(self.feature_cache_dir, exist_ok=True)
            cache_file = os.path.join(
                self.feature_cache_dir,
                self._cache_key(data_loader, subsample_cap) + '.npz')
            if os.path.exists(cache_file):
                cached = np.load(cache_file)
                n_reps = len([k for k in cached.files if k.startswith('rep')])
                return ([
                    torch.from_numpy(cached[f'rep{i}']) for i in range(n_reps)
                ], cached['labels'])

        loader = data_loader
        if subsample_cap is not None and subsample_cap > 0 \
                and len(data_loader.dataset) > subsample_cap:
            generator = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(len(data_loader.dataset),
                                     generator=generator)[:subsample_cap]
            loader = DataLoader(
                data_loader.dataset,
                batch_size=data_loader.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(
                    indices.tolist(), generator=generator),
                num_workers=data_loader.num_workers)

        feats, labels = None, []
        for batch in tqdm(loader, desc=desc, disable=not progress):
            data = batch['data'].cuda()
            batch_feats = self._rep_features(net, data)
            if feats is None:
                feats = [[] for _ in batch_feats]
            for i, f in enumerate(batch_feats):
                feats[i].append(f)
            labels.append(batch['label'])
        feats = [torch.cat(f) for f in feats]
        labels = torch.cat(labels).numpy().astype(int)
        if cache_file is not None:
            np.savez_compressed(
                cache_file, labels=labels,
                **{f'rep{i}': f.cpu().numpy()
                   for i, f in enumerate(feats)})
        return feats, labels

    # ------------------------------------------------------------------
    # per-point PRDC (two-sample statistics)
    # ------------------------------------------------------------------
    def _knn_radii(self, feats: torch.Tensor, k: int):
        """Distance of each point to its kth nearest neighbor in `feats`."""
        n = feats.shape[0]
        k_eff = min(k, n - 1)
        if k_eff < 1:
            return torch.zeros(n, device=feats.device)
        radii = torch.empty(n, device=feats.device)
        for i in range(0, n, self.chunk_size):
            dist = torch.cdist(feats[i:i + self.chunk_size], feats)
            # k_eff + 1 accounts for the zero self-distance
            radii[i:i + self.chunk_size] = dist.kthvalue(k_eff + 1,
                                                         dim=1).values
        return radii

    @torch.no_grad()
    def _prdc_per_point(self, ref: torch.Tensor, ref_radii: torch.Tensor,
                        test: torch.Tensor):
        """Per-point PRDC of `test` (treated as one sample of the two-sample
        test) against the `ref` manifold.

        Returns an (M, 4) tensor with columns (recall, density, precision,
        coverage), the column order of the reference implementation.
        """
        k = self.nearest_k
        n_ref = ref.shape[0]
        test_radii = self._knn_radii(test, k)
        out = torch.empty(test.shape[0], 4, device=test.device)
        for j in range(0, test.shape[0], self.chunk_size):
            chunk = test[j:j + self.chunk_size]
            dist = torch.cdist(ref, chunk)  # (n_ref, c)
            inside_ref = dist < ref_radii[:, None]
            precision = inside_ref.any(dim=0).float()
            density = inside_ref.sum(dim=0).float() / (k * n_ref)
            chunk_radii = test_radii[j:j + self.chunk_size][None, :]
            recall = (dist < chunk_radii).sum(dim=0).float() / n_ref
            coverage = (dist.min(dim=0).values <
                        test_radii[j:j + self.chunk_size]).float()
            out[j:j + self.chunk_size] = torch.stack(
                [recall, density, precision, coverage], dim=1)
        return out

    def _prdc_features(self, feats_list):
        """Concatenated PRDC statistics across representation spaces."""
        assert self.ref_feats is not None, \
            'FortePostprocessor.setup must run before scoring'
        blocks = []
        for ref, ref_radii, test in zip(self.ref_feats, self.ref_radii,
                                        feats_list):
            blocks.append(
                self._prdc_per_point(ref.cuda(), ref_radii.cuda(),
                                     test.cuda()).cpu().numpy())
        return np.concatenate(blocks, axis=1).astype(np.float64)

    # ------------------------------------------------------------------
    # density models over PRDC statistics
    # ------------------------------------------------------------------
    def _fit_density_model(self, fit_prdc: np.ndarray):
        rng = np.random.RandomState(self.seed)
        capped = fit_prdc
        if self.density_fit_cap > 0 and len(fit_prdc) > self.density_fit_cap \
                and self.variant in ('kde', 'ocsvm'):
            idx = rng.choice(len(fit_prdc),
                             self.density_fit_cap,
                             replace=False)
            capped = fit_prdc[idx]

        if self.variant == 'gmm':
            best_model, best_score = None, -np.inf
            for n_components in self.gmm_components:
                if n_components >= len(fit_prdc):
                    continue
                gmm = GaussianMixture(n_components=n_components,
                                      random_state=self.seed)
                gmm.fit(fit_prdc)
                score = gmm.score(fit_prdc)
                if score > best_score:
                    best_model, best_score = gmm, score
            self.density_model = best_model
        elif self.variant == 'kde':
            # PRDC columns are near-degenerate on ID data (binary precision/
            # coverage vs 1e-4-scale recall/density), which makes the KDE
            # covariance ill-conditioned and collapses the whitened log-pdf;
            # a small isotropic jitter regularizes the fit
            data = capped + rng.normal(0, self.kde_jitter, capped.shape)
            kdes = {
                bw: gaussian_kde(data.T, bw_method=bw)
                for bw in ('scott', 'silverman')
            }
            eval_idx = rng.choice(len(data), min(len(data), 2000),
                                  replace=False)
            self.density_model = max(
                kdes.values(),
                key=lambda kde: kde.logpdf(data[eval_idx].T).mean())
        elif self.variant == 'ocsvm':
            best_model, best_ratio = None, -np.inf
            for nu in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
                ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=nu)
                ocsvm.fit(capped)
                ratio = (ocsvm.predict(capped) == 1).mean()
                if ratio > best_ratio:
                    best_model, best_ratio = ocsvm, ratio
            self.density_model = best_model
        else:
            raise ValueError(f'Unknown Forte variant: {self.variant}')

    def _score(self, prdc: np.ndarray):
        """Log-density / decision score of PRDC statistics; higher = ID."""
        if self.variant == 'gmm':
            scores = self.density_model.score_samples(prdc)
        elif self.variant == 'kde':
            scores = np.concatenate([
                self.density_model.logpdf(prdc[i:i + 2000].T)
                for i in range(0, len(prdc), 2000)
            ])
        else:
            scores = self.density_model.decision_function(prdc)
        return np.nan_to_num(np.asarray(scores, dtype=np.float64),
                             nan=-1e12,
                             posinf=1e12,
                             neginf=-1e12).astype(np.float32)

    # ------------------------------------------------------------------
    # OpenOOD interface
    # ------------------------------------------------------------------
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return
        net.eval()
        feats, _ = self._extract_loader(net,
                                        id_loader_dict['train'],
                                        subsample_cap=self.n_setup_samples,
                                        desc='Forte setup: ID features')
        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(feats[0].shape[0], generator=generator)
        half = feats[0].shape[0] // 2
        ref_idx, fit_idx = perm[:half], perm[half:]

        self.ref_feats = [f[ref_idx.to(f.device)].cpu() for f in feats]
        self.ref_radii = [
            self._knn_radii(f.cuda(), self.nearest_k).cpu()
            for f in self.ref_feats
        ]
        self.fit_prdc = self._prdc_features(
            [f[fit_idx.to(f.device)] for f in feats])
        del feats
        torch.cuda.empty_cache()
        self._fit_density_model(self.fit_prdc)
        self.setup_flag = True

    @torch.no_grad()
    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        """Score the whole dataloader as one sample of the two-sample test."""
        net.eval()
        feats, labels = self._extract_loader(net,
                                             data_loader,
                                             progress=progress,
                                             desc='Forte: features')
        conf = self._score(self._prdc_features(feats))
        pred = np.zeros_like(labels)  # Forte is label-free: no classifier
        return pred, conf, labels

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # batch-granular fallback; recall/coverage radii are estimated from
        # the batch, so prefer `inference` on a full dataloader
        feats = self._rep_features(net, data)
        conf = torch.from_numpy(self._score(self._prdc_features(feats)))
        pred = torch.zeros(len(conf), dtype=torch.long)
        return pred, conf

    def set_hyperparam(self, hyperparam: list):
        self.variant = hyperparam[0]
        if self.fit_prdc is not None:
            self._fit_density_model(self.fit_prdc)

    def get_hyperparam(self):
        return self.variant
