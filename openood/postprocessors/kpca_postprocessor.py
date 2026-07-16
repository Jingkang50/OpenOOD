"""
KPCA (Kernel PCA) Postprocessor for Out-of-Distribution Detection.

This module implements the KPCA OOD detection method proposed in:
- Fang et al., "Kernel PCA for Out-of-Distribution Detection", NeurIPS 2024.
- Fang et al., "Kernel PCA for Out-of-Distribution Detection: Non-Linear
  Kernel Selection and Approximation", TPAMI 2026.

The method leverages kernel PCA to learn a non-linear feature subspace from
in-distribution training data, then uses the reconstruction error on this
subspace as the OOD score. Two kernel approximation techniques are supported:
Random Fourier Features (RFF) and Nyström approximation (NYS).
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import scipy
from scipy.special import logsumexp
from sklearn.metrics.pairwise import pairwise_kernels
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class KPCAPostprocessor(BasePostprocessor):
    """Kernel PCA Postprocessor for OOD detection.

    Args:
        config: Config object containing postprocessor settings.
            Expected keys under ``config.postprocessor.postprocessor_args``:
            - approx (str): Kernel approximation method, 'RFF' or 'NYS'.
            - gamma (float): Gaussian kernel bandwidth parameter.
            - M (int): Mapped dimension (RFF) or number of landmarks (NYS).
            - exp_var_ratio (float): Explained variance ratio for PCA
              dimension selection.
            - temperature (float, optional): Temperature for Energy score
              computation used in Nyström landmark selection. Defaults to 1.0.
    """

    def __init__(self, config):
        super(KPCAPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep

        self.approx = self.args.approx
        self.gamma = self.args.gamma
        self.M = self.args.M
        self.exp_var_ratio = self.args.exp_var_ratio
        self.temperature = getattr(self.args, 'temperature', 1.0)

        # Internal state set during setup()
        self.setup_flag = False
        self.ftrain_raw = None
        self.energy_scores = None
        self.u_q = None
        self.q = None
        self.mu = None
        self.w = None
        self.u_rff = None
        self.basis = None
        self.normalization = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """Extract training features and compute KPCA components.

        Extracts penultimate layer features from the in-distribution training
        set, L2-normalizes them (equivalent to a cosine kernel), computes
        Energy scores for Nyström landmark selection, and then applies kernel
        approximation followed by PCA.

        Args:
            net: The classifier network.
            id_loader_dict: Dictionary containing 'train', 'val', 'test'
                data loaders for in-distribution data.
            ood_loader_dict: Dictionary of OOD data loaders (unused).
        """
        if self.setup_flag:
            return

        net.eval()
        features = []
        logits_list = []

        with torch.no_grad():
            for batch in tqdm(id_loader_dict['train'],
                              desc='KPCA setup [feat]:',
                              position=0,
                              leave=True):
                data = batch['data_aux'].cuda()
                logits, feature = net(data, return_feature=True)
                features.append(feature.cpu().numpy())
                logits_list.append(logits.cpu().numpy())

        ftrain = np.concatenate(features, axis=0)
        logits_all = np.concatenate(logits_list, axis=0)

        # L2 normalize (equivalent to Cosine kernel)
        self.ftrain_raw = ftrain / (
            np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10)

        # Compute Energy scores for Nyström landmark selection
        self.energy_scores = self.temperature * logsumexp(
            logits_all / self.temperature, axis=1)

        # Apply kernel approximation and PCA with current hyperparameters
        self._compute_kpca()

        self.setup_flag = True
        print(f'KPCA setup complete: q={self.q}, '
              f'approx={self.approx}, gamma={self.gamma}, M={self.M}, '
              f'exp_var={self.exp_var_ratio}')

    def _compute_kpca(self):
        """Apply kernel approximation and PCA with current hyperparameters.

        This method re-runs the kernel mapping and PCA decomposition using
        the already-extracted raw features. It is called during setup() and
        whenever set_hyperparam() changes the hyperparameters.
        """
        np.random.seed(0) 
        ftrain = self.ftrain_raw.copy()

        # Kernel approximation
        if self.approx == 'RFF':
            ftrain = self._rff_map(ftrain, fit=True)
        elif self.approx == 'NYS':
            ftrain = self._nys_map(ftrain, fit=True)
        else:
            raise ValueError(f'Unknown approximation: {self.approx}')

        # Center the mapped features
        self.mu = ftrain.mean(axis=0)
        ftrain = ftrain - self.mu

        # Linear PCA on kernel-mapped features
        Sigma = ftrain.T.dot(ftrain)
        _, s, vh = scipy.linalg.svd(Sigma)

        # Determine reduced dimension q from explained variance ratio
        q = -1
        s_cumsum = np.cumsum(s)
        s_total = s_cumsum[-1]
        for i in range(len(s)):
            ratio = s_cumsum[i] / s_total
            if i > 0 and q < 0:
                if s_cumsum[i - 1] / s_total < self.exp_var_ratio and ratio >= self.exp_var_ratio:
                    q = i + 1
        if q < 0:
            q = len(s)

        self.q = q
        self.u_q = vh[:q, :].T  # shape: (M, q)

    def _rff_map(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        """Random Fourier Features mapping for Gaussian kernel.

        Approximates the Gaussian kernel K(x,y)=exp(-gamma||x-y||^2) via
        phi(x) = sqrt(2/M) * cos(W·x + u), where W ~ N(0, 2·gamma·I) and
        u ~ Uniform(0, 2π).

        Args:
            x: Input features of shape (N, d).
            fit: If True, generate new random weights; otherwise reuse stored.

        Returns:
            Mapped features of shape (N, M).
        """
        if fit:
            m = x.shape[1]
            self.w = np.sqrt(2 * self.gamma) * np.random.normal(
                size=(int(self.M), m))
            self.u_rff = 2 * np.pi * np.random.rand(int(self.M))
        return np.sqrt(2 / self.M) * np.cos(
            x.dot(self.w.T) + self.u_rff[np.newaxis, :])

    def _nys_map(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        """Nyström low-rank approximation mapping for Gaussian kernel.

        Selects M landmark points with the lowest Energy scores, computes
        the kernel submatrix on landmarks, and derives the low-rank feature
        map via SVD: phi(x) = K(x, landmarks) @ K_MM^{-1/2}.

        Args:
            x: Input features of shape (N, d).
            fit: If True, select landmarks and compute normalization; otherwise
                reuse stored.

        Returns:
            Mapped features of shape (N, M).
        """
        if fit:
            landmark_indices = np.argsort(self.energy_scores)[:int(self.M)]
            self.basis = x[landmark_indices]
            basis_kernel = pairwise_kernels(self.basis,
                                            metric='rbf',
                                            gamma=self.gamma)
            U, S, Vt = scipy.linalg.svd(basis_kernel, full_matrices=True)
            S = np.maximum(S, 1e-12)
            self.normalization = U / np.sqrt(S) @ Vt

        K_xb = pairwise_kernels(x, self.basis, metric='rbf', gamma=self.gamma)
        return K_xb @ self.normalization.T

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """Compute OOD score via KPCA reconstruction error.

        Args:
            net: The classifier network.
            data: Batch of input data.

        Returns:
            Tuple of (pred_class, ood_confidence).
        """
        logits, feature = net(data, return_feature=True)

        # L2 normalize
        feature_np = feature.cpu().numpy()
        feature_np = feature_np / (
            np.linalg.norm(feature_np, axis=-1, keepdims=True) + 1e-10)

        # Kernel approximation
        if self.approx == 'RFF':
            feature_np = self._rff_map(feature_np, fit=False)
        elif self.approx == 'NYS':
            feature_np = self._nys_map(feature_np, fit=False)
        else:
            raise ValueError(f'Unknown approximation: {self.approx}')

        # Center
        feature_np = feature_np - self.mu

        # PCA reconstruction error as OOD score
        # ID samples have smaller reconstruction error → higher (negative) score
        reconstruct = self.u_q.dot(self.u_q.T).dot(feature_np.T).T
        score = -np.linalg.norm(feature_np - reconstruct, axis=1)

        _, pred = torch.max(logits, dim=1)
        return pred, torch.from_numpy(score).float()

    def set_hyperparam(self, hyperparam: list):
        """Set hyperparameters from a list (used by APS mode).

        Args:
            hyperparam: List of [approx, gamma, M, exp_var_ratio].
        """
        self.approx = hyperparam[0]
        self.gamma = hyperparam[1]
        self.M = hyperparam[2]
        self.exp_var_ratio = hyperparam[3]
        if self.setup_flag:
            self._compute_kpca()

    def get_hyperparam(self):
        """Get current hyperparameters as a list.

        Returns:
            List of [approx, gamma, M, exp_var_ratio].
        """
        return [self.approx, self.gamma, self.M, self.exp_var_ratio]
