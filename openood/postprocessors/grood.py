import gc
import math
import time
from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from openood.preprocessors.transform import normalization_dict

from .base_postprocessor import BasePostprocessor

# ---------------------
#   Utils Helpers
# ---------------------


def reduce_memory_usage():
    """Reduces memory usage by clearing CUDA cache and invoking garbage
    collection.

    This function is useful in deep learning models to manage GPU and CPU
    memory.
    """
    # Clear CUDA cache for GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        time.sleep(5)
        torch.cuda.empty_cache()
    # Garbage collection for CPU and any remaining GPU tensors
    gc.collect()


def preprocess_batch(batch, device=None):
    """Preprocess input batch and put it to CUda device.

    Args:
        batch (dict): dictionary or tuple of data and labels

    Returns:
        tuple: a tuple of data and labels
    """
    if type(batch) is dict:
        batch = batch['data'], batch['label']
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch = batch[0].to(device).float(), batch[1].to(device).long()
    return batch


def get_normalization(dataset_name):
    try:
        mean, std = normalization_dict[dataset_name]
    except Exception as e:
        mean = [0.5, 0.5, 0.5]
        std = mean
        print('Could not find normalization for dataset {} giving error '
              '{}'.format(dataset_name, e))
    return mean, std


class NoiseProtoCompute:

    def __init__(
        self,
        network,
        id_loader_dict,
        ood_loader_dict,
        dataset_name,
        prototypes,
        pre_prototypes,
        device,
        pre_layer_indx,
        filter_ood=False,
    ) -> None:
        self.network = network
        self.id_loader_dict = id_loader_dict
        self.ood_loader_dict = ood_loader_dict
        self.dataset_name = dataset_name
        self.device = device
        self.prototypes, self.proto_count = prototypes
        self.pre_prototypes = pre_prototypes
        self.pre_layer_indx = pre_layer_indx
        self.filter_ood = filter_ood
        self.reset()

    def reset(self):
        self.noise_proto = torch.zeros(
            [self.prototypes.shape[-1]],
            device=self.device,
            requires_grad=False,
        )
        self._count_features = torch.zeros(
            [1],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

    def filter_ood_samples(self, feats, threshold_quantile=0.8, min_samples=2):
        """Filter OOD samples while maintaining minimum representation.

        Args:
            feats: OOD feature vectors
            threshold_quantile: Percentile for filtering
            min_samples: Minimum samples to retain

        Returns:
            Filtered features
        """
        distances = torch.cdist(feats, self.prototypes)
        min_distances = distances.min(dim=1)[0]
        # Adaptive threshold based on data distribution
        threshold = torch.quantile(min_distances, threshold_quantile)
        quality_mask = min_distances > threshold

        # Ensure we keep minimum number of samples
        if quality_mask.sum() < min_samples:
            # Keep top-k most distant samples
            k = min(min_samples, len(feats))
            _, indices = torch.topk(min_distances, k)
            quality_mask = torch.zeros_like(quality_mask)
            quality_mask[indices] = True

        return feats[quality_mask]

    def update_noise_proto(self, feats):
        """Updates the noise prototype as a running mean of the features.

        Args:
            feats (Tensor): input features used to update noise prototype
        """
        n_features = feats.shape[0]
        if self.filter_ood:
            feats = self.filter_ood_samples(feats)
            n_features = feats.shape[0]
            if n_features == 0:
                return
        feats_sum = torch.sum(feats, dim=0)
        self.noise_proto = (feats_sum +
                            self._count_features[0] * self.noise_proto) / (
                                self._count_features[0] + n_features)
        self._count_features[0] += n_features

    @torch.no_grad()
    def get_uniform_noise(self, n, inp_channel, img_size):
        """Updates OOD prototype using uniform noisde images.

        Args:
            n (int): number of images to be generated
            inp_channel (int): number of channels of generated images
            img_size (tuple): tuple of width and height of generated images
        """
        mean, std = get_normalization(self.dataset_name)
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(self.device)
        pbar = tqdm(
            total=n,
            desc='Computing Uniform Noise Prototype',
            leave=True,
        )
        bsz = 100
        actual_bsz = 10
        n_iters = int(math.ceil(n / bsz))
        for _ in range(n_iters):
            noise_imgs = torch.rand(
                (bsz, inp_channel, img_size, img_size)).to(self.device)
            # smoothes the neighbor of the uniform image
            noise_imgs = torchvision.transforms.GaussianBlur(
                kernel_size=(5, 7), sigma=(0.2, 5))(noise_imgs)

            logits, features = self.network(noise_imgs,
                                            return_feature_list=True)
            # selecting noise imgs having the lowest energys core
            energyconf = -1 * torch.logsumexp(logits, dim=1).view(-1)
            # high energy score means ID while low energy score means OOD
            # we need to sample data that have similar properties as that of ID
            top_samples = torch.topk(energyconf, actual_bsz,
                                     dim=0).indices.view(-1)
            penultimate_feature = features[-1][top_samples]
            self.update_noise_proto(penultimate_feature.contiguous().view(
                actual_bsz, -1))
            pbar.update(bsz)

    @torch.no_grad()
    def get_val_ood(self, n):
        """Use Validation OOD to generate an OOD prototype.

        Args:
            n (int): number of samples to use
        """
        pbar = tqdm(
            total=n,
            desc='Computing Val OOD Noise Prototype',
            leave=True,
        )
        count_features = 0
        for ood_data in self.ood_loader_dict['val']:
            real_ood = preprocess_batch(ood_data, device=self.device)[0]
            bsz = real_ood.shape[0]
            _, features_ood = self.network(real_ood, return_feature_list=True)
            self.update_noise_proto(features_ood[-1].contiguous().view(
                bsz, -1))
            pbar.update(bsz)
            count_features += bsz
            if count_features >= n:
                break

    @torch.no_grad()
    def get_prototype_mean(self, ):
        """Computes mean of ID prototypes as our noise prototype."""
        print('Computing Mean Prototype')
        self._count_features[0] += self.prototypes.shape[0]
        self.noise_proto = self.prototypes.clone().mean(0).view(-1)

    @torch.no_grad()
    def get_synthetic_ood(self, n):
        """Generate syntthettic OOD data using mixup on ID data.

        Args:
            n (int): number of samples tto generate
        """
        pbar = tqdm(
            total=n,
            desc='Computing Synthetic ID Noise Prototype',
            leave=True,
        )
        count_features = 0

        for id_data in self.id_loader_dict['train']:
            real_id, _ = preprocess_batch(id_data, self.device)
            # maybe apply a data transformation that converts ID data to OOD
            # apply channel rgb shuffling
            bsz = real_id.shape[0]
            logits, perturbed_features = self.network(real_id,
                                                      return_feature_list=True)

            # Original mixup logic
            nearest_k_targets = torch.topk(logits, 2, dim=1).indices[:, 1:]
            pre_features = perturbed_features[self.pre_layer_indx]

            # Reshape pre_features for prototype mixing
            if pre_features.ndim == 2:
                new_shape = (1, pre_features.shape[1])
            elif pre_features.ndim == 3:
                new_shape = (
                    1,
                    pre_features.shape[1],
                    pre_features.shape[2],
                )
            else:
                new_shape = (
                    1,
                    pre_features.shape[1],
                    pre_features.shape[2],
                    pre_features.shape[3],
                )
            pre_prototypes = [
                proto.view(new_shape) for proto in self.pre_prototypes
            ]
            pre_features = perturbed_features[self.pre_layer_indx].clone()
            perturbed_features = torch.zeros_like(perturbed_features[-1])
            for i in range(bsz):
                mixup_lam = 0.5
                tmp_logits, tmp_feats = get_mixed_feats(
                    self.network,
                    pre_features[i].unsqueeze(0),
                    self.pre_layer_indx,
                    mixup_lam,  # Fixed mixup ratio
                    1,
                    pre_prototypes,
                    nearest_k_targets[i].unsqueeze(0),
                )
                perturbed_features[i] = (
                    tmp_feats.reshape_as(perturbed_features[i])
                    # + torch.randn_like(perturbed_features[i]) * 0.05
                )
                tmp_logits = tmp_logits.reshape(1, logits.shape[1])
                # now we need to measure its OODness
                print('****************************************************')
                print('Mixing lam used {}'.format(mixup_lam))
                print('MSP score for generated OOD {:.4f}'.format(
                    tmp_logits.softmax(dim=1).max(1).values.item()))
                print('MSP score for ID {:.4f}'.format(logits[i].view(
                    1, logits.shape[1]).softmax(dim=1).max(1).values.item()))
                top2_logits = tmp_logits.topk(2, dim=1)[0]
                logit_margin = (top2_logits[:, 0] - top2_logits[:, 1]).item()
                print('Logit margin between top two classes {:.4f}'.format(
                    logit_margin))
                print('****************************************************')
            bsz = real_id.shape[0]
            perturbed_features = perturbed_features.contiguous().view(bsz, -1)
            self.update_noise_proto(perturbed_features)
            count_features += bsz
            pbar.update(bsz)

            if count_features >= n:
                break

        pbar.close()


# -------------------------
#   BaseModel wrapper
# -------------------------
class BaseModel(nn.Module):
    """Model wrapper for all models to divide it into sub-layers."""

    def __init__(self, net: nn.Module):
        super(BaseModel, self).__init__()
        self.model = net
        self.model.eval()
        self.fix_bn()
        # index layers to map from feature index to model layer index
        self.layers_map = {}
        self.flatten = (False if hasattr(self.model, 'no_flatten')
                        and self.model.no_flatten else True)
        if hasattr(self.model, 'get_layer_list'):
            self.layers = self.model.get_layer_list()
        else:
            self.layers = [layer for layer in self.model.children()]

    def index_layers(self, random_img):
        """Indexing layers from feature index to layer index.

        Args:
            random_img (Tensor): input random image to create the map
        """
        _, features = self.forward(random_img, return_feature_list=True)
        # bug in ResNet-18 model layers are not matching with layer
        x = random_img
        for layer_indx, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear) and self.flatten:
                x = x.view(x.shape[0], -1)
            x = layer(x)
            for feat_indx, feat in enumerate(features):
                if x.shape == feat.shape and torch.isclose(x, feat).all():
                    self.layers_map[feat_indx] = layer_indx + 1
                    break

    def get_penultimate_dim(self, input_size, device, layer_index=None):
        """Returns penultimate dimension.

        Args:
            input_size (int): size of the input image Width=Height.
            device (str): device cpu or gpu
            layer_index (int, optional): index of the penultimate layer.
                Defaults to None.

        Returns:
            int: penultimate feature dimension
        """
        random_img = torch.rand(1, 3, input_size, input_size, device=device)
        self.index_layers(random_img)
        feature = self.get_penultimate(random_img, layer_index=layer_index)
        return feature.flatten().shape[-1]

    def get_penultimate(self, x, layer_index=None):
        """Returns penultimate output for input x.

        Args:
            x (tensor): input tensor
            layer_index (int, optional): layer idex of the penultimate layer.
                Defaults to None.

        Returns:
            tensor: penultimate layer output tensor
        """
        _, feature_list = self.forward(x, return_feature_list=True)
        if layer_index is None:
            return feature_list[-1]
        return feature_list[layer_index]

    def intermediate_forward(self,
                             x,
                             layer_start=0,
                             layer_end=-1,
                             return_logits=False):
        """Forward pass from an intermediate representation.

        Args:
            x (tensor): input representation
            layer_start (int, optional): index of the layer start.
                Defaults to 0.
            layer_end (int, optional): index of layer end defaults to
                penultimate layer. Defaults to -1.
            return_logits (bool, optional): flag to return logits or not.
                Defaults to False.

        Returns:
            tensor: intermediate representation output or tuple with both
            logits and representation based on return_logits.
        """
        layers = self.layers
        if layer_end < 0:
            layer_end = len(layers) + layer_end - 1
        # map from layer start to the model's layer
        layer_start = self.layers_map[layer_start]
        stop_criteria = len(layers) if return_logits else layer_end
        intermediate_out = None
        for i in range(layer_start, stop_criteria):
            if isinstance(layers[i], nn.Linear) and self.flatten:
                # flatten operation
                x = x.view(x.shape[0], -1)
            x = layers[i].forward(x)
            if i == layer_end:
                intermediate_out = x
        if return_logits:
            return x, intermediate_out
        return x

    def forward(self, x, return_feature_list=False):
        """Forward pass over the model.

        Args:
            x (tensor): input image
            return_feature_list (bool, optional): flag to return intermediate
                features as a list. Defaults to False.

        Returns:
            tensor/tuple: output of the model or a tuple with feature list
            based on input flag
        """
        return self.model.forward(x, return_feature_list=return_feature_list)

    def fix_bn(self):
        """Used to fix batch norm and ABN norms."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.training = False


# ------------------------------------------
# Loss Utils to return interpolated features
# ------------------------------------------


def get_mixed_feats(
    network,
    pre_features,
    layer_mix,
    lam,
    k,
    pre_prototypes,
    top_k_targets,
):
    bsz, *feat_shape = pre_features.shape
    feats_weighted = lam * pre_features

    cl_unique = top_k_targets.unique()
    proto_weighted = (1 - lam) * torch.cat(
        [pre_prototypes[cl] for cl in cl_unique], dim=0)

    # Fix: Create index mapping from cl_unique instead of proto_weighted
    indx_map = {cl.item(): cl_indx for cl_indx, cl in enumerate(cl_unique)}

    mapped_top_k_targets = (torch.tensor([
        indx_map[t.item()] for t in top_k_targets.flatten()
    ]).reshape(top_k_targets.shape).to(top_k_targets.device))

    all_mixed_input = feats_weighted.unsqueeze(1) + proto_weighted

    expanded_indices = mapped_top_k_targets.view(
        -1, k, *([1] * len(feat_shape))).expand(-1, -1, *feat_shape)

    mixed_inputs = torch.gather(all_mixed_input, 1,
                                expanded_indices).view(-1, *feat_shape)
    mixed_logits, mixed_feats = network.intermediate_forward(
        mixed_inputs,
        layer_start=layer_mix,
        layer_end=-1,
        return_logits=True,
    )

    return mixed_logits, mixed_feats.reshape(bsz * k, -1)


# -----------------------
#   Method Code
# -----------------------


class GrOODPostprocessor(BasePostprocessor):

    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        # index of the layer where we do the mixup
        self.pre_layer_indx = self.args.pre_layer_indx
        # mixup lambda usually higher than 0.7 to give more weight
        # to the representation of the input data point
        self.lam = self.args.lam
        # parameter for the number of synthetic OOD samples
        self.n_noise_samples = self.args.n_noise_samples
        # flag to use direct distance loss instead of the cross entropy
        # on the distance defaults to False
        self.distance_loss = (True if hasattr(self.args, 'distance_loss')
                              and self.args.distance_loss else False)
        # flag to use validation set instead of the training for the
        # computation of prototypes and indexing of gradients.
        self.use_val = (True if hasattr(self.args, 'use_val')
                        and self.args.use_val else False)
        # flag to include gaussian noise in the computation of the noise
        # prototype. defaults to True
        self.use_gaussian_noise = (True
                                   if hasattr(self.args, 'use_gaussian_noise')
                                   and self.args.use_gaussian_noise else False)
        # flag to include validation OOD in the computation of noise centroid
        self.use_val_ood = (True if hasattr(self.args, 'use_val_ood')
                            and self.args.use_val_ood else False)
        # flag to include Synthetic OOD in the computation of noise centroid
        self.use_synthetic = (True if hasattr(self.args, 'use_synthetic')
                              and self.args.use_synthetic else False)
        self.synthetic_name = (self.args.synthetic_name if hasattr(
            self.args, 'synthetic_name') else 'synthetic')
        # flag to use mean of prototypes as our prototype computation
        self.use_proto_mean = (True if hasattr(self.args, 'use_proto_mean')
                               and self.args.use_proto_mean else False)
        # flag to use cosine similarity instead of euclidean distance in Faiss
        self.use_cosine = (True if hasattr(self.args, 'use_cosine')
                           and self.args.use_cosine else False)
        # flag to use pseudo prototype gradients instead of the noise gradients
        self.use_pseudo_prototype = (
            True if hasattr(self.args, 'use_pseudo_prototype')
            and self.args.use_pseudo_prototype else False)
        # flag to use distances to the noise prototype directly instead of
        # computing and indexing gradients
        self.no_gradients = (True if hasattr(self.args, 'no_gradients')
                             and self.args.no_gradients else False)
        # flag to simply use the norm of the gradients instead of knn indexing
        self.use_grad_norm = (True if hasattr(self.args, 'use_grad_norm')
                              and self.args.use_grad_norm else False)
        self.use_direct_grad = (True if hasattr(self.args, 'use_direct_grad')
                                and self.args.use_direct_grad else False)
        self.filter_ood = (True if hasattr(self.args, 'filter_ood')
                           and self.args.filter_ood else False)
        # k nearest points distance defaults to 1
        self.k = self.args.k
        # ratio / percentage of the dataset to be used for indexing
        self.sampling_ratio = self.args.sampling_ratio
        # init prototypes objects
        self._pre_prototypes_tensors = None
        self._count_features = None
        self._prototypes_tensors = None
        self.network = None
        # when enabled we index and compute on a single batch
        # only to make sure it works
        self.debug = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """Called at the start of the preprocessor to setup centroids and faiss
        index.

        Args:
            net (nn.Module): input network
            id_loader_dict (dict): ID dataloader dictionary
                including train and val.
            ood_loader_dict (dict): OOD dataloader including val set.
        """
        # Initialize network, datamodule and other related variables
        self.network = BaseModel(net)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.network.fix_bn()
        self.network.eval()

        # on_fit_start
        # get penultimate layer dimension
        penultimate_dim = self.network.get_penultimate_dim(
            self.config.dataset.image_size,
            self.device,
        )

        num_classes = self.config.dataset.num_classes
        dataset_name = self.config.dataset.name
        # on_test_start
        if self.use_val:
            train_dataloader = id_loader_dict['val']
        else:
            train_dataloader = id_loader_dict['train']
        # now computing class prototypes
        # initialize prototypes
        self._prototypes_tensors = self._init_prototypes(
            num_classes, penultimate_dim, self.device)
        # get intermediate layer dimension
        pre_layer_dim = self.network.get_penultimate_dim(
            self.config.dataset.image_size,
            self.device,
            layer_index=self.pre_layer_indx,
        )
        self._pre_prototypes_tensors = self._init_prototypes(
            num_classes, pre_layer_dim, self.device)
        n_samples = self.sampling_ratio * len(train_dataloader)
        self.n_samples = n_samples
        for idx, batch in tqdm(
                enumerate(train_dataloader),
                desc='Eval: update prototype',
                position=0,
                leave=True,
        ):
            batch = preprocess_batch(batch, self.device)
            self.update_prototype(batch)
            if self.debug or idx >= n_samples:
                break
        # generate the OOD prototype
        self.noise_proto_compute = NoiseProtoCompute(
            self.network,
            id_loader_dict,
            ood_loader_dict,
            dataset_name,
            (self.prototypes, self._count_features),
            self.pre_prototypes,
            device=self.device,
            pre_layer_indx=self.pre_layer_indx,
            filter_ood=self.filter_ood,
        )
        if self.use_gaussian_noise:
            self.noise_proto_compute.get_uniform_noise(
                self.n_noise_samples, 3, self.config.dataset.image_size)
        if self.use_val_ood:
            self.noise_proto_compute.get_val_ood(self.n_noise_samples)
        if self.use_synthetic:
            if self.synthetic_name == 'synthetic':
                self.noise_proto_compute.get_synthetic_ood(
                    self.n_noise_samples)
            else:
                print('Synthetic OOD name not recognized {}'.format(
                    self.synthetic_name))
        if self.use_proto_mean:
            self.noise_proto_compute.get_prototype_mean()
        self.noise_proto = self.noise_proto_compute.noise_proto.detach()
        reduce_memory_usage()

        if self.no_gradients or self.use_grad_norm:
            return
        self.id_loader_dict = id_loader_dict
        self.index_grads()
        mean, std = get_normalization(self.dataset_name)
        self.mean = mean
        self.std = std

    def index_grads(self):
        train_dataloader = self.id_loader_dict['train']
        if self.use_cosine:
            quantizer = faiss.IndexFlatIP(self.penultimate_dim)
        else:
            quantizer = faiss.IndexFlatL2(self.penultimate_dim)
        # setting number of cluster to 100 regardless of the number of samples
        # as it won't throw an exception and in that case for anything
        # fewer than 100 it will run similar to flat index
        indexed_grads = []
        # just training on 80% of the samples or 600 as a min.
        train_index_samples = min(600, int(self.n_samples * 0.8))
        batch = preprocess_batch(next(iter(train_dataloader)), self.device)
        # nlist = min(int(self.n_samples * batch[0].shape[0] // 39), 100)
        nlist = int(4 * math.sqrt(train_index_samples))
        self.index = faiss.IndexIVFFlat(
            quantizer,
            self.penultimate_dim,
            nlist,
            (faiss.METRIC_INNER_PRODUCT
             if self.use_cosine else faiss.METRIC_L2),
        )
        index_trained = False
        # import time  <-- This was the unused import

        for idx, batch in tqdm(
                enumerate(train_dataloader),
                desc='Eval: Indexing gradients',
                position=0,
                leave=True,
        ):
            grads = self._compute_grad(batch)['grads']
            if index_trained:
                self.index.add(grads)
            elif idx >= train_index_samples:
                all_grads = np.concatenate(indexed_grads, axis=0)
                self.index.train(all_grads)
                self.index.add(all_grads)
                del all_grads
                index_trained = True
            else:
                indexed_grads.append(grads)
            if self.debug or idx >= self.n_samples:
                break
        # makes faiss faster by specifying number of cells to visit
        # TODO we might need to tweak it for our method
        self.index.nprobe = 4

    def compute_ncc_loss(self, img):
        """Custom loss for which we compute the gradients.

        Args:
            img (Tensor): input batch data

        Returns:
            tuple: losses including original and mixup nearest centroid loss
            beside the pseudo label used
        """
        prototypes = self.prototypes
        noise_proto = self.noise_proto
        logits, feature_list = self.network(img, return_feature_list=True)
        pseudo_preds = logits.argmax(1).view(-1).long()
        bsz = logits.shape[0]
        original_feature = feature_list[-1]
        n_prototypes = len(prototypes)
        prototypes = prototypes.view(n_prototypes, -1)
        prototypes = torch.cat([prototypes, noise_proto.view(1, -1)], dim=0)
        original_vector_prototypes = (
            original_feature.reshape(bsz, 1, -1).detach() -
            prototypes).reshape(bsz, n_prototypes + 1, -1)
        ncc_scores = -1 * torch.norm(
            original_vector_prototypes,
            dim=2,
        )
        # make it predict target class vs others
        # make mixed preds predict everything else except that class
        losses = []
        if self.distance_loss:
            losses.append(ncc_scores.mean(-1).view(bsz))
        elif self.no_gradients and self.use_pseudo_prototype:
            losses.append(ncc_scores[torch.arange(bsz), pseudo_preds.long()])
        elif self.no_gradients:
            losses.append(ncc_scores[:, -1])
        else:
            losses.append(
                F.cross_entropy(
                    ncc_scores,
                    pseudo_preds,
                    reduction='none',
                ))
        # is it a good idea to add simclr loss??
        return {
            'losses': losses,
            'pseudo_preds': pseudo_preds,
            'ncc_scores': ncc_scores,
            'original_vector_prototypes': original_vector_prototypes,
            'penultimate_feature': original_feature.reshape(bsz, -1),
        }

    @torch.no_grad()
    def _compute_grad_directly(self, batch, data_alone=False):
        """Computes the gradients wrt class centroids.

        Args:
            batch (tuple): input batch including data and labels
            data_alone (bool, optional): flag to read only data coming from
                the batch in case of inference. Defaults to False.

        Returns:
            np.array: numpy array of gradients computed
        """
        if data_alone:
            data = batch.to(self.device).float()
        else:
            batch = preprocess_batch(batch, self.device)
            # views
            data, labels = batch
        bsz = data.shape[0]
        # also choice of lam is crucial for the results lam=0.9 k= 3
        loss_data = self.compute_ncc_loss(data, )
        pseudo_preds = loss_data['pseudo_preds']
        ncc_scores = loss_data['ncc_scores'].unsqueeze(-1)
        vectors_from_noise = loss_data['original_vector_prototypes'][:, -1, :]
        # normalize vectors
        unit_vectors_from_noise = F.normalize(vectors_from_noise)
        proba_noise = ncc_scores.softmax(-1)[:, -1]
        grads = proba_noise * unit_vectors_from_noise
        grads = grads.view(bsz, -1)
        if not (data_alone):
            # filter out gradients to use only correctly classified ones
            grads = grads.view(bsz, -1)[labels == pseudo_preds]
        return {
            'grads': grads.cpu().numpy(),
            'pseudo_preds': pseudo_preds,
            'ncc_preds': loss_data['ncc_scores'][:, :-1].argmax(1),
            'penultimate_feature': loss_data['penultimate_feature'],
        }

    def _compute_grad(self, batch, data_alone=False):
        """Computes the gradients wrt class centroids.

        Args:
            batch (tuple): input batch including data and labels
            data_alone (bool, optional): flag to read only data coming from
                the batch in case of inference. Defaults to False.

        Returns:
            np.array: numpy array of gradients computed
        """
        if self.use_direct_grad:
            return self._compute_grad_directly(batch, data_alone=data_alone)
        if data_alone:
            data = batch.to(self.device).float()
        else:
            batch = preprocess_batch(batch, device=self.device)
            # views
            data, labels = batch
        with torch.enable_grad():
            bsz = data.shape[0]
            self._prototypes_tensors.requires_grad = True
            self.noise_proto.requires_grad = True
            loss_data = self.compute_ncc_loss(data, )
            losses = loss_data['losses']
            pseudo_preds = loss_data['pseudo_preds']
            ncc_preds = loss_data['ncc_scores'][:, :-1].argmax(1)
            penultimate_feature = loss_data['penultimate_feature']
            del loss_data
            grads = []
            loss = losses[0]
            if self.no_gradients:
                return {
                    'grads': loss.detach().cpu().numpy(),
                    'pseudo_preds': pseudo_preds,
                    'ncc_preds': ncc_preds,
                    'penultimate_feature': penultimate_feature,
                }
            for i in range(bsz):
                if not (data_alone) and labels[i] != pseudo_preds[i]:
                    continue
                # add condition if the data point is correctly classified
                gradients_aug = torch.autograd.grad(
                    loss[i],
                    [self.noise_proto, self.prototypes],
                    retain_graph=True,
                )
                gradients = gradients_aug[0].data.view(1, -1)
                gradients_proto = (gradients_aug[1].data[pseudo_preds[i]].view(
                    1, -1))
                if self.use_pseudo_prototype:
                    gradients = gradients_proto
                grads.append(gradients)
            self._prototypes_tensors.requires_grad = False
            self.noise_proto.requires_grad = False
            return {
                'grads': torch.concatenate(grads, dim=0).cpu().numpy(),
                'pseudo_preds': pseudo_preds,
                'ncc_preds': ncc_preds,
                'penultimate_feature': penultimate_feature,
            }

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """Post processing on input data to return confidence.

        Args:
            net (nn.Module): input data model
            data (torch.tensor): input tensor to compute preds and confidence
                on it

        Returns:
            tuple: predictions and confidence score.
        """
        # score should be high for ID and low for OOD
        grads_data = self._compute_grad(data, data_alone=True)
        grads = grads_data['grads']
        pred = grads_data['ncc_preds']
        if self.no_gradients and self.use_pseudo_prototype:
            score_ood = grads
        elif self.no_gradients:
            score_ood = -1 * grads
        elif self.use_grad_norm:
            score_ood = -1 * np.sum(np.abs(grads), axis=1)
        else:
            dists, _ = self.index.search(grads, 1)
            score_ood = np.stack(-1 * dists[:, -1])
        score_ood = torch.from_numpy(score_ood).to(pred.device)
        # our method returns NCC predictions for reporting in the paper itself
        return pred, score_ood

    # Prototype related functions
    @property
    def pre_prototypes(self):
        return self._pre_prototypes_tensors

    @property
    def prototypes(self):
        return self._prototypes_tensors

    def _init_prototypes(self, n_classes, penultimate_dim, device):
        """Initializes the prototypes at start of each task
        Args:
            task_num (int): task number
            accelerator (Trainer.accelerator): pytorch lightning
                accelerator used
            penultimate_dim (int): dimension size of penultimate layer
        """
        proto_tensors = torch.zeros(
            [n_classes, penultimate_dim],
            device=device,
            requires_grad=False,
        )
        self._count_features = torch.zeros(
            [n_classes],
            dtype=torch.long,
            device=device,
            requires_grad=False,
        )
        proto_tensors.requires_grad = False
        self._count_features.requires_grad = False
        return proto_tensors

    def are_prototypes_ready(self):
        """Detects if at least a data point per class seen so far
        Returns:
            bool: flag to denote if all our prototypes are nonzeros
        """
        return (self._count_features is not None
                and self._count_features.count_nonzero()
                == self.self._count_features.shape[0])

    def update_prototype(self, batch):
        """Updates existing prototypes based on data and labels.

        Args:
            batch (tuple): tuple of input data and labels.
        """

        def get_total_features(feats_cl, cl, proto_tensor):
            feats_cl_sum = torch.sum(feats_cl, dim=0)
            return (feats_cl_sum + self._count_features[cl] *
                    proto_tensor[cl]) / (self._count_features[cl] + n_features)

        data, target = batch
        _, features = self.network(data, return_feature_list=True)
        pre_features = (features[self.pre_layer_indx].detach().reshape(
            data.shape[0], -1))
        # for the prototypes
        self.penultimate_dim = features[-1].shape[1]
        features = features[-1].detach().reshape(data.shape[0], -1)
        classes = target.unique()
        for cl in classes:
            cl = cl.long()
            masked_cl = (target == cl).bool()
            n_features = masked_cl.sum()
            if n_features == 0:
                continue
            self._pre_prototypes_tensors[cl] = get_total_features(
                pre_features[masked_cl],
                cl,
                self._pre_prototypes_tensors,
            )
            self._prototypes_tensors[cl] = get_total_features(
                features[masked_cl], cl, self._prototypes_tensors)
            self._count_features[cl] += n_features

    def set_hyperparam(self, hyperparam: list):
        self.pre_layer_indx = hyperparam[0]
        self.lam = hyperparam[1]
        self.k = hyperparam[2]
        self.sampling_ratio = hyperparam[3]
        self.use_gaussian_noise = hyperparam[4]
        self.use_val_ood = hyperparam[5]

    def get_hyperparam(self):
        return [
            self.pre_layer_indx,
            self.lam,
            self.k,
            self.sampling_ratio,
            self.use_gaussian_noise,
            self.use_val_ood,
        ]
