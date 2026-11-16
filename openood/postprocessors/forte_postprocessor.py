import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
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
    evaluated sample rather than of single samples. Following the reference
    protocol, ID test and OOD data are fused into ONE evaluated sample per
    OOD dataset before computing PRDC (`inference_fused`); at deployment
    time the test stream contains both distributions, so estimating the
    recall/coverage radii in the mixture is the unbiased choice. The
    `fused_two_sample` attribute tells the evaluator to use this pairwise
    path. Per-dataloader `inference` is deliberately unsupported: it would
    change the evaluated sample (biasing recall/coverage) and, under the
    default protocol, score the reference and fit samples themselves.

    `setup` follows the reference implementation's data protocol
    (main.py, github.com/DebarghaG/forte). With the default
    `reference_pool: test`, the ID evaluation pool is split 67/33 into an
    ID-train part and a held-out part; the ID-train part is split 50/50
    into the reference manifold and the density-model fit set, and only
    the held-out third is evaluated (fused with each OOD dataset). The
    reference sample is therefore drawn from the same pool as the
    evaluated ID data, which a two-sample test requires. With
    `reference_pool: train`, an external ID train sample provides
    reference and fit sets (50/50) and the full ID test set is evaluated
    -- OpenOOD's usual setup semantics, but distribution shift between
    train and test images then costs accuracy.

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
        self.kde_jitter = getattr(self.args, 'kde_jitter', 1e-3)
        # 'test': paper protocol -- reference/fit/held-out all split from
        # the ID evaluation pool. 'train': external ID train reference.
        self.reference_pool = getattr(self.args, 'reference_pool', 'test')
        self.heldout_fraction = getattr(self.args, 'heldout_fraction', 0.33)
        # optional directory for caching extracted features across runs
        self.feature_cache_dir = getattr(self.args, 'feature_cache_dir', None)
        # no hyperparameter is tuned on OOD validation data
        self.args_dict = {}
        self.APS_mode = False
        # tells evaluators that ID and OOD must be scored jointly per pair
        self.fused_two_sample = True
        self.setup_flag = False
        self.ref_feats = None
        self.ref_radii = None
        self.fit_prdc = None
        self.density_model = None
        # held-out ID sample (paper protocol): the evaluated ID side
        self.heldout_feats = None
        self.heldout_labels = None
        # in-memory memo of extracted ID features: the fused protocol
        # re-evaluates the ID sample once per OOD dataset
        self._feat_memo = {}
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

    @staticmethod
    def _is_subsampled(dataset, subsample_cap):
        return subsample_cap is not None and subsample_cap > 0 \
            and len(dataset) > subsample_cap

    def _cache_key(self, net: nn.Module, data_loader: DataLoader,
                   subsample_cap):
        dataset = data_loader.dataset
        name = getattr(dataset, 'name', type(dataset).__name__)
        # cached features are only valid for the representation spaces that
        # produced them, so the network identity must be part of the key
        net_sig = '-'.join(getattr(net, 'backbone_names',
                                   [type(net).__name__]))
        # the seed only influences which samples are drawn when subsampling
        suffix = (f'cap{subsample_cap}_seed{self.seed}' if self._is_subsampled(
            dataset, subsample_cap) else 'all')
        return f'{name}_{len(dataset)}_{net_sig}_{suffix}'

    @torch.no_grad()
    def _extract_loader(self,
                        net: nn.Module,
                        data_loader: DataLoader,
                        subsample_cap: int = None,
                        progress: bool = True,
                        desc: str = 'Feature extraction',
                        memo: bool = False):
        memo_key = None
        if memo:
            memo_key = (id(data_loader.dataset), len(data_loader.dataset),
                        subsample_cap)
            if memo_key in self._feat_memo:
                return self._feat_memo[memo_key]

        cache_file = None
        if self.feature_cache_dir:
            os.makedirs(self.feature_cache_dir, exist_ok=True)
            cache_file = os.path.join(
                self.feature_cache_dir,
                self._cache_key(net, data_loader, subsample_cap) + '.npz')
            if os.path.exists(cache_file):
                cached = np.load(cache_file)
                n_reps = len([k for k in cached.files if k.startswith('rep')])
                result = ([
                    torch.from_numpy(cached[f'rep{i}']) for i in range(n_reps)
                ], cached['labels'])
                if memo_key is not None:
                    self._feat_memo[memo_key] = result
                return result

        loader = data_loader
        if self._is_subsampled(data_loader.dataset, subsample_cap):
            generator = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(len(data_loader.dataset),
                                     generator=generator)[:subsample_cap]
            loader = DataLoader(data_loader.dataset,
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
                # keep the accumulator on CPU: the full feature matrix does
                # not need to live on the GPU next to the backbones
                feats[i].append(f.cpu())
            labels.append(batch['label'])
        if feats is None:
            raise ValueError(
                f'{desc}: dataloader over '
                f'{type(loader.dataset).__name__} yielded no batches; '
                'is the dataset empty?')
        feats = [torch.cat(f) for f in feats]
        labels = torch.cat(labels).numpy().astype(int)
        if cache_file is not None:
            np.savez(cache_file,
                     labels=labels,
                     **{
                         f'rep{i}': f.cpu().numpy()
                         for i, f in enumerate(feats)
                     })
        if memo_key is not None:
            self._feat_memo[memo_key] = (feats, labels)
        return feats, labels

    # ------------------------------------------------------------------
    # per-point PRDC (two-sample statistics)
    # ------------------------------------------------------------------
    def _knn_radii(self, feats: torch.Tensor, k: int):
        """Distance of each point to its kth nearest neighbor in `feats`."""
        # float64: torch.cdist's float32 matmul path loses the distance
        # differences PRDC thresholds on when feature norms dwarf the k-NN
        # radii (ViT-MSN: norm ~400 vs radius ~18), flipping neighbor counts
        # relative to the reference implementation's float64 distances
        feats = feats.double()
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
        # float64 for the same reason as in `_knn_radii`
        ref, ref_radii, test = ref.double(), ref_radii.double(), test.double()
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
            coverage = (dist.min(dim=0).values
                        < test_radii[j:j + self.chunk_size]).float()
            out[j:j + self.chunk_size] = torch.stack(
                [recall, density, precision, coverage], dim=1)
        return out

    def _prdc_features(self, feats_list):
        """Concatenated PRDC statistics across representation spaces."""
        assert self.ref_feats is not None, \
            'FortePostprocessor.setup must run before scoring'
        if len(feats_list) != len(self.ref_feats):
            raise ValueError(
                f'got {len(feats_list)} representation spaces but the '
                f'reference was built with {len(self.ref_feats)}; '
                'a stale feature cache from a different backbone set is '
                'the usual cause')
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
            if best_model is None:
                raise ValueError(
                    f'no GMM candidate in {list(self.gmm_components)} has '
                    f'fewer components than the {len(fit_prdc)} fit samples; '
                    'the ID train sample is too small')
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
            eval_idx = rng.choice(len(data),
                                  min(len(data), 2000),
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

        use_test_pool = (self.reference_pool == 'test'
                         and 'test' in id_loader_dict)
        pool_key = 'test' if use_test_pool else 'train'
        feats, labels = self._extract_loader(
            net,
            id_loader_dict[pool_key],
            subsample_cap=self.n_setup_samples,
            desc=f'Forte setup: ID {pool_key} pool features')

        idx = np.arange(feats[0].shape[0])
        if use_test_pool:
            # reference protocol (main.py): pool -> 67/33 id-train/held-out,
            # id-train -> 50/50 reference/fit; only the held-out part is
            # evaluated, so the reference comes from the same pool as the
            # evaluated ID sample
            train_idx, held_idx = train_test_split(
                idx, test_size=self.heldout_fraction, random_state=self.seed)
            self.heldout_feats = [f[held_idx] for f in feats]
            self.heldout_labels = labels[held_idx]
            print(
                f'Forte setup (paper protocol): pool {len(idx)} -> '
                f'reference+fit {len(train_idx)}, '
                f'held-out ID {len(held_idx)}',
                flush=True)
        else:
            # external ID train sample: all of it feeds reference/fit and
            # the full ID test set is evaluated
            train_idx = idx
            self.heldout_feats = None
            self.heldout_labels = None
        ref_idx, fit_idx = train_test_split(train_idx,
                                            test_size=0.5,
                                            random_state=self.seed)

        self.ref_feats = [f[ref_idx] for f in feats]
        self.ref_radii = [
            self._knn_radii(f.cuda(), self.nearest_k).cpu()
            for f in self.ref_feats
        ]
        self.fit_prdc = self._prdc_features([f[fit_idx] for f in feats])
        del feats
        torch.cuda.empty_cache()
        self._fit_density_model(self.fit_prdc)
        self.setup_flag = True

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        raise NotImplementedError(
            'Forte is a two-sample test and cannot score one dataloader at '
            'a time: per-point recall/coverage depend on the k-NN radii of '
            'the whole evaluated sample, and under the default protocol the '
            'ID test loader also contains the reference and fit samples, '
            'which must not be scored. Evaluate through '
            '`openood.evaluation_api.Evaluator` (see '
            'scripts/eval_ood_forte.py), which calls `inference_fused` for '
            'every ID/OOD pair. Automatic hyperparameter search (APS) and '
            'the config-based test pipelines are not supported.')

    @torch.no_grad()
    def inference_fused(self,
                        net: nn.Module,
                        id_loaders,
                        ood_loader: DataLoader,
                        progress: bool = True):
        """Score ID and OOD jointly as ONE fused evaluated sample (the
        reference protocol).

        The recall/coverage k-NN radii are estimated within the fused
        ID + OOD sample, mirroring deployment where the test stream mixes
        both distributions. Because the fused set differs per OOD dataset,
        the ID confidences returned here are pair-specific.

        Args:
            id_loaders: list of ID dataloaders, the first being the ID
                test set (plus csid sets for full-spectrum evaluation);
                their samples form the ID side of the fused sample. Under
                the paper protocol (`reference_pool: test`) the ID test
                loader is represented by the held-out split stored during
                setup instead of being re-extracted.
            ood_loader: the OOD dataloader forming the other side.

        Returns:
            (id_pred, id_conf, id_gt, ood_pred, ood_conf, ood_gt)
        """
        net.eval()
        if self.heldout_feats is not None:
            # paper protocol: the evaluated ID sample is the held-out
            # split of the setup pool (the first loader in id_loaders)
            id_feats = [[f] for f in self.heldout_feats]
            id_labels = [self.heldout_labels]
            extra_loaders = id_loaders[1:]
        else:
            id_feats, id_labels = None, []
            extra_loaders = id_loaders
        for loader in extra_loaders:
            feats, labels = self._extract_loader(net,
                                                 loader,
                                                 progress=progress,
                                                 desc='Forte: ID features',
                                                 memo=True)
            if id_feats is None:
                id_feats = [[f] for f in feats]
            else:
                for i, f in enumerate(feats):
                    id_feats[i].append(f)
            id_labels.append(labels)
        id_feats = [torch.cat(f) for f in id_feats]
        id_labels = np.concatenate(id_labels)

        ood_feats, ood_labels = self._extract_loader(
            net, ood_loader, progress=progress, desc='Forte: OOD features')

        n_id = id_feats[0].shape[0]
        fused = [
            torch.cat([f_id, f_ood])
            for f_id, f_ood in zip(id_feats, ood_feats)
        ]
        conf = self._score(self._prdc_features(fused))
        id_conf, ood_conf = conf[:n_id], conf[n_id:]
        return (-np.ones_like(id_labels), id_conf, id_labels,
                -np.ones_like(ood_labels), ood_conf, ood_labels)

    def postprocess(self, net: nn.Module, data: Any):
        raise NotImplementedError(
            'Forte is a two-sample test: recall/coverage are statistics of '
            'the whole evaluated sample, so batch-granular scoring is '
            'ill-defined. Use `inference_fused` (ID + OOD pair).')
