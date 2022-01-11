import os
import pickle
import sys

import numpy as np
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from sklearn.covariance import (empirical_covariance, ledoit_wolf,
                                shrunk_covariance)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm

to_np = lambda x: x.data.cpu().numpy()


def get_torch_feature_stat(feature, only_mean=False):
    feature = feature.view([feature.size(0), feature.size(1), -1])
    feature_mean = torch.mean(feature, dim=-1)
    feature_var = torch.var(feature, dim=-1)
    if feature.size(-2) * feature.size(-1) == 1 or only_mean:
        # [N, C, 1, 1] does not need variance for kernel
        feature_stat = feature_mean
    else:
        feature_stat = torch.cat((feature_mean, feature_var), 1)
    return feature_stat


def process_feature_type(feature_temp, feature_type):
    if feature_type == 'flat':
        feature_temp = feature_temp.view(
            [feature_temp.size(0),
             feature_temp.size(1), -1])
    elif feature_type == 'stat':
        feature_temp = get_torch_feature_stat(feature_temp)
    elif feature_type == 'mean':
        feature_temp = get_torch_feature_stat(feature_temp, only_mean=True)
    else:
        raise ValueError('Unknown feature type')
    return feature_temp


def reduce_feature_dim(feature_list_full, label_list_full, feature_process):
    if feature_process == 'none':
        transform_matrix = np.eye(feature_list_full.shape[1])
    else:
        feature_process, kept_dim = feature_process.split('_')
        kept_dim = int(kept_dim)
        if feature_process == 'capca':
            lda = InverseLDA(solver='eigen')
            lda.fit(feature_list_full, label_list_full)
            transform_matrix = lda.scalings_[:, :kept_dim]
        elif feature_process == 'pca':
            pca = PCA(n_components=kept_dim)
            pca.fit(feature_list_full)
            transform_matrix = pca.components_.T
        elif feature_process == 'lda':
            lda = LinearDiscriminantAnalysis(solver='eigen')
            lda.fit(feature_list_full, label_list_full)
            transform_matrix = lda.scalings_[:, :kept_dim]
        else:
            raise Exception('Unknown Process Type')
    return transform_matrix


def sample_estimator(train_loader, model, num_classes, feature_type_list,
                     feature_process_list):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    model.eval()
    num_layer = len(feature_type_list)
    feature_mean_list = [[None] * num_classes] * num_layer
    # generate a list of features
    for layer_idx in range(num_layer):
        feature_type = feature_type_list[layer_idx]
        feature_process = feature_process_list[layer_idx]
        feature_list_full, label_list_full = [], []
        with torch.no_grad():
            for batch in train_loader:
                data = batch['plain_data'].cuda()
                label = batch['label']
                _, feature_list = model(data, return_feature_list=True)
                feature_list = process_feature_type(feature_list[layer_idx],
                                                    feature_type)
                feature_list_full.extend(to_np(feature_list))
                label_list_full.extend(to_np(label))
        feature_list_full = np.array(feature_list_full)
        label_list_full = np.array(label_list_full)
        transform_matrix = reduce_feature_dim(feature_list_full,
                                              label_list_full, feature_process)
        new_feature_list = np.dot(feature_list_full, transform_matrix)
        for feature, label in zip(new_feature_list, label_list_full):
            if feature_mean_list[layer_idx][label] == None:
                feature_mean_list[layer_idx][label] = feature
            else:
                feature_mean_list[layer_idx][label].append(feature)
        import pdb
        pdb.set_trace()
    precision_list = []
    for k in range(num_output):
        X = []
        for i in range(num_classes):
            X.append(list_features[k][i] - sample_class_mean[k][i])
        X = torch.cat(X, 0)

        # find inverse
        group_lasso.fit(X.numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    sample_class_mean = [t.cuda() for t in sample_class_mean]
    return sample_class_mean, precision


def get_Mahalanobis_score(model, test_loader, num_classes, sample_mean,
                          precision, layer_index, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    for batch in tqdm(
            test_loader,
            desc=f'val_{test_loader.dataset.name}_layer{layer_index}'):
        data = batch['data'].cuda()
        data = Variable(data, requires_grad=True)
        noise_gaussian_score = compute_noise_Mahalanobis_score(
            model, data, num_classes, sample_mean, precision, layer_index,
            magnitude)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
    return Mahalanobis


def compute_noise_Mahalanobis_score(model,
                                    data,
                                    num_classes,
                                    sample_mean,
                                    precision,
                                    layer_index,
                                    magnitude,
                                    return_pred=False):
    _, out_features = model(data, return_feature=True)
    out_features = out_features[layer_index]
    out_features = out_features.view(out_features.size(0),
                                     out_features.size(1), -1)
    out_features = torch.mean(out_features, 2)

    # compute Mahalanobis score
    gaussian_score = 0
    for i in range(num_classes):
        batch_sample_mean = sample_mean[layer_index][i]
        zero_f = out_features.data - batch_sample_mean
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]),
                                   zero_f.t()).diag()
        if i == 0:
            gaussian_score = term_gau.view(-1, 1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)),
                                       1)

    # Input_processing
    sample_pred = gaussian_score.max(1)[1]
    batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
    zero_f = out_features - Variable(batch_sample_mean)
    pure_gau = -0.5 * torch.mm(
        torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
    loss = torch.mean(-pure_gau)
    loss.backward()

    gradient = torch.ge(data.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # here we use the default value of 0.5
    gradient.index_copy_(
        1,
        torch.LongTensor([0]).cuda(),
        gradient.index_select(1,
                              torch.LongTensor([0]).cuda()) / 0.5)
    gradient.index_copy_(
        1,
        torch.LongTensor([1]).cuda(),
        gradient.index_select(1,
                              torch.LongTensor([1]).cuda()) / 0.5)
    gradient.index_copy_(
        1,
        torch.LongTensor([2]).cuda(),
        gradient.index_select(1,
                              torch.LongTensor([2]).cuda()) / 0.5)
    tempInputs = torch.add(
        data.data, gradient,
        alpha=-magnitude)  # updated input data with perturbation

    with torch.no_grad():
        _, noise_out_features = model(Variable(tempInputs),
                                      return_feature=True)
        noise_out_features = noise_out_features[layer_index]
        noise_out_features = noise_out_features.view(
            noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)

    noise_gaussian_score = 0
    for i in range(num_classes):
        batch_sample_mean = sample_mean[layer_index][i]
        zero_f = noise_out_features.data - batch_sample_mean
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]),
                                   zero_f.t()).diag()
        if i == 0:
            noise_gaussian_score = term_gau.view(-1, 1)
        else:
            noise_gaussian_score = torch.cat(
                (noise_gaussian_score, term_gau.view(-1, 1)), 1)

    noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
    if return_pred:
        return sample_pred, noise_gaussian_score
    else:
        return noise_gaussian_score


def alpha_selector(data_in, data_out):
    label_in = np.ones(len(data_in))
    label_out = np.zeros(len(data_out))
    data = np.concatenate([data_in, data_out])
    label = np.concatenate([label_in, label_out])
    # skip the last-layer flattened feature (duplicated with the last feature)
    lr = LogisticRegressionCV(n_jobs=-1).fit(data, label)
    alpha_list = lr.coef_.reshape(-1)
    print(f'Optimal Alpha List: {alpha_list}')
    return alpha_list


def _cov(X, shrinkage=None, covariance_estimator=None):
    """Estimate covariance matrix (using optional covariance_estimator).
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    shrinkage : {'empirical', 'auto'} or float, default=None
        Shrinkage parameter, possible values:
          - None or 'empirical': no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.
        Shrinkage parameter is ignored if  `covariance_estimator`
        is not None.
    covariance_estimator : estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying on the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in :mod:`sklearn.covariance``.
        if None the shrinkage parameter drives the estimate.
        .. versionadded:: 0.24
    Returns
    -------
    s : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.
    """
    if covariance_estimator is None:
        shrinkage = 'empirical' if shrinkage is None else shrinkage
        if isinstance(shrinkage, str):
            if shrinkage == 'auto':
                sc = StandardScaler()  # standardize features
                X = sc.fit_transform(X)
                s = ledoit_wolf(X)[0]
                # rescale
                s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
            elif shrinkage == 'empirical':
                s = empirical_covariance(X)
            else:
                raise ValueError('unknown shrinkage parameter')
        elif isinstance(shrinkage, float) or isinstance(shrinkage, int):
            if shrinkage < 0 or shrinkage > 1:
                raise ValueError('shrinkage parameter must be between 0 and 1')
            s = shrunk_covariance(empirical_covariance(X), shrinkage)
        else:
            raise TypeError('shrinkage must be a float or a string')
    else:
        if shrinkage is not None and shrinkage != 0:
            raise ValueError('covariance_estimator and shrinkage parameters '
                             'are not None. Only one of the two can be set.')
        covariance_estimator.fit(X)
        if not hasattr(covariance_estimator, 'covariance_'):
            raise ValueError('%s does not have a covariance_ attribute' %
                             covariance_estimator.__class__.__name__)
        s = covariance_estimator.covariance_
    return s


def _class_means(X, y):
    """Compute class means.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    Returns
    -------
    means : array-like of shape (n_classes, n_features)
        Class means.
    """
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]
    return means


def _class_cov(X, y, priors, shrinkage=None, covariance_estimator=None):
    """Compute weighted within-class covariance matrix.
    The per-class covariance are weighted by the class priors.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    priors : array-like of shape (n_classes,)
        Class priors.
    shrinkage : 'auto' or float, default=None
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.
        Shrinkage parameter is ignored if `covariance_estimator` is not None.
    covariance_estimator : estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in sklearn.covariance.
        If None, the shrinkage parameter drives the estimate.
        .. versionadded:: 0.24
    Returns
    -------
    cov : array-like of shape (n_features, n_features)
        Weighted within-class covariance matrix
    """
    classes = np.unique(y)
    cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    for idx, group in enumerate(classes):
        Xg = X[y == group, :]
        cov += priors[idx] * np.atleast_2d(
            _cov(Xg, shrinkage, covariance_estimator))
    return cov


class InverseLDA(LinearDiscriminantAnalysis):
    def _solve_eigen(self, X, y, shrinkage):
        """Eigenvalue solver.
        The eigenvalue solver computes the optimal solution of the Rayleigh
        coefficient (basically the ratio of between class scatter to within
        class scatter). This solver supports both classification and
        dimensionality reduction (with optional shrinkage).
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        shrinkage : string or float, optional
            Shrinkage parameter, possible values:
              - None: no shrinkage (default).
              - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
              - float between 0 and 1: fixed shrinkage constant.
        Notes
        -----
        This solver is based on [1]_, section 3.8.3, pp. 121-124.
        References
        ----------
        [1] Pattern Classification (Second Edition).
            John Wiley & Sons, Inc., New York, 2001. ISBN
        """
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(X, y, self.priors_, shrinkage)

        Sw = self.covariance_  # within scatter
        St = _cov(X, shrinkage)  # total scatter
        Sb = St - Sw  # between scatter

        # Standard LDA: evals, evecs = linalg.eigh(Sb, Sw)
        # Here we hope to find a mapping to maximize Sw with minimum Sb for class agnostic.
        evals, evecs = linalg.eigh(Sw)

        self.explained_variance_ratio_ = np.sort(
            evals / np.sum(evals))[::-1][:self._max_components]
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = (-0.5 * np.diag(np.dot(self.means_, self.coef_.T)) +
                           np.log(self.priors_))
