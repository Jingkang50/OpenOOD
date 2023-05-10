import math
import numbers

import diffdist.functional as distops
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Function
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config


class CSITrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:
        self.net = net['backbone']
        self.train_loader = train_loader
        self.config = config
        self.mode = config.mode

        if self.config.num_gpus > 1:
            self.dummy_net = net['dummy_net'].module
        else:
            self.dummy_net = net['dummy_net']
        self.dummy_net.cpu()

        self.simclr_aug = get_simclr_augmentation(
            config, image_size=config.dataset.image_size).cuda()
        self.linear = net['linear']
        self.linear_optim = torch.optim.Adam(
            self.linear.parameters(),
            lr=1e-3,
            betas=(.9, .999),
            weight_decay=config.optimizer.weight_decay)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.hflip = HorizontalFlipLayer().cuda()

        self.simclr_layer = net['simclr_layer']
        self.rotation_linear = net['shift_cls_layer']
        self.joint_linear = net['joint_distribution_layer']

        if 'step1' in self.mode:
            self.optimizer = optim.SGD(
                list(self.net.parameters()) +
                list(self.simclr_layer.parameters()),
                lr=config.optimizer.lr,
                momentum=0.9,
                weight_decay=config.optimizer.weight_decay)
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, config.optimizer.num_epochs)
            self.scheduler_warmup = GradualWarmupScheduler(
                self.optimizer,
                multiplier=10.0,
                total_epoch=config.optimizer.warmup,
                after_scheduler=self.scheduler)
        else:
            milestones = [
                int(0.6 * config.optimizer.num_epochs),
                int(0.75 * config.optimizer.num_epochs),
                int(0.9 * config.optimizer.num_epochs)
            ]

            self.linear_optim = torch.optim.Adam(
                self.linear.parameters(),
                lr=1e-3,
                betas=(.9, .999),
                weight_decay=config.optimizer.weight_decay)
            self.linear_scheduler = lr_scheduler.MultiStepLR(
                self.linear_optim, gamma=0.1, milestones=milestones)

            self.rotation_linear_optim = torch.optim.SGD(
                self.rotation_linear.parameters(),
                lr=1e-1,
                weight_decay=config.optimizer.weight_decay)
            self.rot_scheduler = lr_scheduler.MultiStepLR(
                self.rotation_linear_optim, gamma=0.1, milestones=milestones)

            self.joint_linear_optim = torch.optim.SGD(
                self.joint_linear.parameters(),
                lr=1e-1,
                weight_decay=config.optimizer.weight_decay)
            self.joint_scheduler = lr_scheduler.MultiStepLR(
                self.joint_linear_optim, gamma=0.1, milestones=milestones)

    def train_epoch(self, epoch_idx):
        if 'step1' in self.mode:
            return self.train_sup_epoch(epoch_idx)
        else:
            return self.train_suplinear_epoch(epoch_idx)

    def train_sup_epoch(self, epoch_idx):
        self.net.train()
        train_dataiter = iter(self.train_loader)

        n = 0
        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            images = batch['data'].cuda()
            labels = batch['label'].cuda()

            batch_size = images.size(0)
            images1, images2 = self.hflip(images.repeat(2, 1, 1,
                                                        1)).chunk(2)  # hflip
            images1 = torch.cat(
                [torch.rot90(images1, rot, (2, 3)) for rot in range(4)])  # 4B
            images2 = torch.cat(
                [torch.rot90(images2, rot, (2, 3)) for rot in range(4)])  # 4B
            images_pair = torch.cat([images1, images2], dim=0)  # 8B

            rot_sim_labels = torch.cat([
                labels + self.config.dataset.num_classes * i for i in range(4)
            ],
                                       dim=0)

            images_pair = self.simclr_aug(images_pair)  # simclr augment
            _, features = self.net(images_pair, return_feature=True)

            simclr_outputs = self.simclr_layer(features)
            simclr = normalize(simclr_outputs)  # normalize
            sim_matrix = get_similarity_matrix(
                simclr, multi_gpu=self.config.num_gpus > 1)
            loss_sim = Supervised_NT_xent(
                sim_matrix,
                labels=rot_sim_labels,
                temperature=self.config.temperature,
                multi_gpu=self.config.num_gpus > 1) * self.config.sim_lambda

            # total loss
            loss = loss_sim

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.scheduler.step(epoch_idx - 1 + n / len(self.train_loader))
            # lr = self.optimizer.param_groups[0]['lr']

            # Post-processing stuffs
            penul_1 = features[:batch_size]
            penul_2 = features[4 * batch_size:5 * batch_size]
            features = torch.cat([penul_1,
                                  penul_2])  # only use original rotation

            # Linear evaluation
            outputs_linear_eval = self.linear(features.detach())
            loss_linear = self.criterion(outputs_linear_eval, labels.repeat(2))

            self.linear_optim.zero_grad()
            loss_linear.backward()
            self.linear_optim.step()

            n = n + 1

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss

        if self.config.num_gpus > 1:
            self.dummy_net.backbone.load_state_dict(
                self.net.module.state_dict())
            self.dummy_net.linear.load_state_dict(
                self.linear.module.state_dict())
            self.dummy_net.simclr_layer.load_state_dict(
                self.simclr_layer.module.state_dict())
            self.dummy_net.joint_distribution_layer.load_state_dict(
                self.joint_distribution_layer.module.state_dict())
            self.dummy_net.shift_cls_layer.load_state_dict(
                self.shift_cls_layer.module.state_dict())
        else:
            self.dummy_net.backbone.load_state_dict(self.net.state_dict())
            self.dummy_net.linear.load_state_dict(self.linear.state_dict())
            self.dummy_net.simclr_layer.load_state_dict(
                self.simclr_layer.state_dict())
            self.dummy_net.joint_distribution_layer.load_state_dict(
                self.joint_distribution_layer.state_dict())
            self.dummy_net.shift_cls_layer.load_state_dict(
                self.shift_cls_layer.state_dict())

        return self.dummy_net, metrics

    def train_suplinear_epoch(self, epoch_idx):
        self.net.train()
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            self.net.eval()
            batch = next(train_dataiter)
            images = batch['data'].cuda()
            labels = batch['label'].cuda()

            batch_size = images.size(0)
            images = self.hflip(images)
            images = torch.cat(
                [torch.rot90(images, rot, (2, 3)) for rot in range(4)])  # 4B
            rot_labels = torch.cat(
                [torch.ones_like(labels) * k for k in range(4)], 0)  # B -> 4B
            joint_labels = torch.cat([
                labels + self.config.dataset.num_classes * i for i in range(4)
            ],
                                     dim=0)

            images = self.simclr_aug(images)  # simclr augmentation
            _, features = self.net(images, return_feature=True)
            penultimate = features.detach()

            outputs = self.linear(
                penultimate[0:batch_size]
            )  # only use 0 degree samples for linear eval
            outputs_rot = self.rotation_linear(penultimate)
            outputs_joint = self.joint_linear(penultimate)

            loss_ce = self.criterion(outputs, labels)
            loss_rot = self.criterion(outputs_rot, rot_labels)
            loss_joint = self.criterion(outputs_joint, joint_labels)

            # CE loss
            self.linear_optim.zero_grad()
            loss_ce.backward()
            self.linear_optim.step()

            # Rot loss
            self.rotation_linear_optim.zero_grad()
            loss_rot.backward()
            self.rotation_linear_optim.step()

            # Joint loss
            self.joint_linear_optim.zero_grad()
            loss_joint.backward()
            self.joint_linear_optim.step()

            # optimizer learning rate
            # lr = self.linear_optim.param_groups[0]['lr']

        self.linear_scheduler.step()
        self.rot_scheduler.step()
        self.joint_scheduler.step()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_ce + loss_rot + loss_joint

        if self.config.num_gpus > 1:
            self.dummy_net.backbone.load_state_dict(
                self.net.module.state_dict())
            self.dummy_net.linear.load_state_dict(
                self.linear.module.state_dict())
            self.dummy_net.simclr_layer.load_state_dict(
                self.simclr_layer.module.state_dict())
            self.dummy_net.joint_distribution_layer.load_state_dict(
                self.joint_distribution_layer.module.state_dict())
            self.dummy_net.shift_cls_layer.load_state_dict(
                self.shift_cls_layer.module.state_dict())
        else:
            self.dummy_net.backbone.load_state_dict(self.net.state_dict())
            self.dummy_net.linear.load_state_dict(self.linear.state_dict())
            self.dummy_net.simclr_layer.load_state_dict(
                self.simclr_layer.state_dict())
            self.dummy_net.joint_distribution_layer.load_state_dict(
                self.joint_distribution_layer.state_dict())
            self.dummy_net.shift_cls_layer.load_state_dict(
                self.shift_cls_layer.state_dict())

        return self.dummy_net, metrics


def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    """Compute similarity matrix.

    - outputs: (B', d) tensor for B' = B * chunk
    - sim_matrix: (B', B') tensor
    """

    if multi_gpu:
        outputs_gathered = []
        for out in outputs.chunk(chunk):
            gather_t = [
                torch.empty_like(out) for _ in range(dist.get_world_size())
            ]
            gather_t = torch.cat(distops.all_gather(gather_t, out))
            outputs_gathered.append(gather_t)
        outputs = torch.cat(outputs_gathered)

    sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')

    return sim_matrix


def Supervised_NT_xent(sim_matrix,
                       labels,
                       temperature=0.5,
                       chunk=2,
                       eps=1e-8,
                       multi_gpu=False):
    """Compute NT_xent loss.

    - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    """

    device = sim_matrix.device

    if multi_gpu:
        gather_t = [
            torch.empty_like(labels) for _ in range(dist.get_world_size())
        ]
        labels = torch.cat(distops.all_gather(gather_t, labels))
    labels = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye
                                                        )  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    labels = labels.contiguous().view(-1, 1)
    Mask = torch.eq(labels, labels.t()).float().to(device)
    # Mask = eye * torch.stack([labels == labels[i]
    # for i in range(labels.size(0))]).float().to(device)
    Mask = Mask / (Mask.sum(dim=1, keepdim=True) + eps)

    loss = torch.sum(Mask * sim_matrix) / (2 * B)

    return loss


def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def get_simclr_augmentation(config, image_size):

    # parameter for resizecrop
    resize_scale = (config.resize_factor, 1.0)  # resize scaling factor
    if config.resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (config.resize_factor, config.resize_factor)

    # Align augmentation
    color_jitter = ColorJitterLayer(brightness=0.4,
                                    contrast=0.4,
                                    saturation=0.4,
                                    hue=0.1,
                                    p=0.8)
    color_gray = RandomColorGrayLayer(p=0.2)
    resize_crop = RandomResizedCropLayer(scale=resize_scale, size=image_size)

    # Transform define #
    if config.dataset.name == 'imagenet':
        # Using RandomResizedCrop at PIL transform
        transform = nn.Sequential(
            color_jitter,
            color_gray,
        )
    else:
        transform = nn.Sequential(
            color_jitter,
            color_gray,
            resize_crop,
        )

    return transform


# ----------Warmup Scheduler----------
class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer. Proposed in
    'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        if multiplier > 1.0. if multiplier = 1.0,
        lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch,
        use this scheduler (eg. ReduceLROnPlateau)
    """
    def __init__(self,
                 optimizer,
                 multiplier,
                 total_epoch,
                 after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError(
                'multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr *
                ((self.multiplier - 1.) * self.last_epoch / self.total_epoch +
                 1.) for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1
        # ReduceLROnPlateau is called at the end of epoch,
        # whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr *
                ((self.multiplier - 1.) * self.last_epoch / self.total_epoch +
                 1.) for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


# ----------transform layers----------
if torch.__version__ >= '1.4.0':
    kwargs = {'align_corners': False}
else:
    kwargs = {}


def rgb2hsv(rgb):
    """Convert a 4-d RGB tensor to the HSV counterpart.

    Here, we compute hue using atan2() based on the definition in [1],
    instead of using the common lookup table approach as in [2, 3].
    Those values agree when the angle is a multiple of 30°,
    otherwise they may differ at most ~1.2°.

    References
    [1] https://en.wikipedia.org/wiki/Hue
    [2] https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    [3] https://github.com/scikit-image/scikit-image/
    blob/master/skimage/color/colorconv.py#L212
    """

    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

    Cmax = rgb.max(1)[0]
    Cmin = rgb.min(1)[0]
    delta = Cmax - Cmin

    hue = torch.atan2(math.sqrt(3) * (g - b), 2 * r - g - b)
    hue = (hue % (2 * math.pi)) / (2 * math.pi)
    saturate = delta / Cmax
    value = Cmax
    hsv = torch.stack([hue, saturate, value], dim=1)
    hsv[~torch.isfinite(hsv)] = 0.
    return hsv


def hsv2rgb(hsv):
    """Convert a 4-d HSV tensor to the RGB counterpart.

    >>> %timeit hsv2rgb(hsv)
    2.37 ms ± 13.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    >>> %timeit rgb2hsv_fast(rgb)
    298 µs ± 542 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    >>> torch.allclose(hsv2rgb(hsv), hsv2rgb_fast(hsv), atol=1e-6)
    True

    References
    [1] https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB_alternative
    """
    h, s, v = hsv[:, [0]], hsv[:, [1]], hsv[:, [2]]
    c = v * s

    n = hsv.new_tensor([5, 3, 1]).view(3, 1, 1)
    k = (n + h * 6) % 6
    t = torch.min(k, 4 - k)
    t = torch.clamp(t, 0, 1)

    return v - c * t


class RandomResizedCropLayer(nn.Module):
    def __init__(self, size=None, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        """Inception Crop size (tuple): size of forwarding image (C, W, H)
        scale (tuple): range of size of the origin size cropped ratio (tuple):

        range of aspect ratio of the origin aspect ratio cropped.
        """
        super(RandomResizedCropLayer, self).__init__()

        _eye = torch.eye(2, 3)
        self.size = size
        self.register_buffer('_eye', _eye)
        self.scale = scale
        self.ratio = ratio

    def forward(self, inputs, whbias=None):
        _device = inputs.device
        N = inputs.size(0)
        _theta = self._eye.repeat(N, 1, 1)

        if whbias is None:
            whbias = self._sample_latent(inputs)

        _theta[:, 0, 0] = whbias[:, 0]
        _theta[:, 1, 1] = whbias[:, 1]
        _theta[:, 0, 2] = whbias[:, 2]
        _theta[:, 1, 2] = whbias[:, 3]

        grid = F.affine_grid(_theta, inputs.size(), **kwargs).to(_device)
        output = F.grid_sample(inputs,
                               grid,
                               padding_mode='reflection',
                               **kwargs)

        if self.size is not None:
            output = F.adaptive_avg_pool2d(output, self.size)

        return output

    def _clamp(self, whbias):

        w = whbias[:, 0]
        h = whbias[:, 1]
        w_bias = whbias[:, 2]
        h_bias = whbias[:, 3]

        # Clamp with scale
        w = torch.clamp(w, *self.scale)
        h = torch.clamp(h, *self.scale)

        # Clamp with ratio
        w = self.ratio[0] * h + torch.relu(w - self.ratio[0] * h)
        w = self.ratio[1] * h - torch.relu(self.ratio[1] * h - w)

        # Clamp with bias range: w_bias \in (w - 1, 1 - w),
        # h_bias \in (h - 1, 1 - h)
        w_bias = w - 1 + torch.relu(w_bias - w + 1)
        w_bias = 1 - w - torch.relu(1 - w - w_bias)

        h_bias = h - 1 + torch.relu(h_bias - h + 1)
        h_bias = 1 - h - torch.relu(1 - h - h_bias)

        whbias = torch.stack([w, h, w_bias, h_bias], dim=0).t()

        return whbias

    def _sample_latent(self, inputs):

        _device = inputs.device
        N, _, width, height = inputs.shape

        # N * 10 trial
        area = width * height
        target_area = np.random.uniform(*self.scale, N * 10) * area
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        aspect_ratio = np.exp(np.random.uniform(*log_ratio, N * 10))

        # If doesn't satisfy ratio condition, then do central crop
        w = np.round(np.sqrt(target_area * aspect_ratio))
        h = np.round(np.sqrt(target_area / aspect_ratio))
        cond = (0 < w) * (w <= width) * (0 < h) * (h <= height)
        w = w[cond]
        h = h[cond]
        cond_len = w.shape[0]
        if cond_len >= N:
            w = w[:N]
            h = h[:N]
        else:
            w = np.concatenate([w, np.ones(N - cond_len) * width])
            h = np.concatenate([h, np.ones(N - cond_len) * height])

        w_bias = np.random.randint(w - width, width - w + 1) / width
        h_bias = np.random.randint(h - height, height - h + 1) / height
        w = w / width
        h = h / height

        whbias = np.column_stack([w, h, w_bias, h_bias])
        whbias = torch.tensor(whbias, device=_device)

        return whbias


class HorizontalFlipRandomCrop(nn.Module):
    def __init__(self, max_range):
        super(HorizontalFlipRandomCrop, self).__init__()
        self.max_range = max_range
        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)

    def forward(self, input, sign=None, bias=None, rotation=None):
        _device = input.device
        N = input.size(0)
        _theta = self._eye.repeat(N, 1, 1)

        if sign is None:
            sign = torch.bernoulli(torch.ones(N, device=_device) * 0.5) * 2 - 1
        if bias is None:
            bias = torch.empty(
                (N, 2), device=_device).uniform_(-self.max_range,
                                                 self.max_range)
        _theta[:, 0, 0] = sign
        _theta[:, :, 2] = bias

        if rotation is not None:
            _theta[:, 0:2, 0:2] = rotation

        grid = F.affine_grid(_theta, input.size(), **kwargs).to(_device)
        output = F.grid_sample(input,
                               grid,
                               padding_mode='reflection',
                               **kwargs)

        return output

    def _sample_latent(self, N, device=None):
        sign = torch.bernoulli(torch.ones(N, device=device) * 0.5) * 2 - 1
        bias = torch.empty(
            (N, 2), device=device).uniform_(-self.max_range, self.max_range)
        return sign, bias


class Rotation(nn.Module):
    def __init__(self, max_range=4):
        super(Rotation, self).__init__()
        self.max_range = max_range
        self.prob = 0.5

    def forward(self, input, aug_index=None):

        _, _, H, W = input.size()

        if aug_index is None:
            aug_index = np.random.randint(4)

            output = torch.rot90(input, aug_index, (2, 3))

            _prob = input.new_full((input.size(0), ), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1 - _mask) * output

        else:
            aug_index = aug_index % self.max_range
            output = torch.rot90(input, aug_index, (2, 3))

        return output


class CutPerm(nn.Module):
    def __init__(self, max_range=4):
        super(CutPerm, self).__init__()
        self.max_range = max_range
        self.prob = 0.5

    def forward(self, input, aug_index=None):

        _, _, H, W = input.size()

        if aug_index is None:
            aug_index = np.random.randint(4)

            output = self._cutperm(input, aug_index)

            _prob = input.new_full((input.size(0), ), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1 - _mask) * output

        else:
            aug_index = aug_index % self.max_range
            output = self._cutperm(input, aug_index)

        return output

    def _cutperm(self, inputs, aug_index):

        _, _, H, W = inputs.size()
        h_mid = int(H / 2)
        w_mid = int(W / 2)

        jigsaw_h = aug_index // 2
        jigsaw_v = aug_index % 2

        if jigsaw_h == 1:
            inputs = torch.cat(
                (inputs[:, :, h_mid:, :], inputs[:, :, 0:h_mid, :]), dim=2)
        if jigsaw_v == 1:
            inputs = torch.cat(
                (inputs[:, :, :, w_mid:], inputs[:, :, :, 0:w_mid]), dim=3)

        return inputs


class HorizontalFlipLayer(nn.Module):
    def __init__(self):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(HorizontalFlipLayer, self).__init__()

        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)

    def forward(self, inputs):
        _device = inputs.device

        N = inputs.size(0)
        _theta = self._eye.repeat(N, 1, 1)
        r_sign = torch.bernoulli(torch.ones(N, device=_device) * 0.5) * 2 - 1
        _theta[:, 0, 0] = r_sign
        grid = F.affine_grid(_theta, inputs.size(), **kwargs).to(_device)
        inputs = F.grid_sample(inputs,
                               grid,
                               padding_mode='reflection',
                               **kwargs)

        return inputs


class RandomColorGrayLayer(nn.Module):
    def __init__(self, p):
        super(RandomColorGrayLayer, self).__init__()
        self.prob = p

        _weight = torch.tensor([[0.299, 0.587, 0.114]])
        self.register_buffer('_weight', _weight.view(1, 3, 1, 1))

    def forward(self, inputs, aug_index=None):

        if aug_index == 0:
            return inputs

        outputs = F.conv2d(inputs, self._weight)
        gray = torch.cat([outputs, outputs, outputs], dim=1)

        if aug_index is None:
            _prob = inputs.new_full((inputs.size(0), ), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)

            gray = inputs * (1 - _mask) + gray * _mask

        return gray


class ColorJitterLayer(nn.Module):
    def __init__(self, p, brightness, contrast, saturation, hue):
        super(ColorJitterLayer, self).__init__()
        self.prob = p
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue,
                                     'hue',
                                     center=0,
                                     bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self,
                     value,
                     name,
                     center=1,
                     bound=(0, float('inf')),
                     clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    'If {} is a single number, it must be non negative.'.
                    format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError('{} values should be between {}'.format(
                    name, bound))
        else:
            raise TypeError(
                '{} should be a single number or a list/tuple with length 2.'.
                format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def adjust_contrast(self, x):
        if self.contrast:
            factor = x.new_empty(x.size(0), 1, 1, 1).uniform_(*self.contrast)
            means = torch.mean(x, dim=[2, 3], keepdim=True)
            x = (x - means) * factor + means
        return torch.clamp(x, 0, 1)

    def adjust_hsv(self, x):
        f_h = x.new_zeros(x.size(0), 1, 1)
        f_s = x.new_ones(x.size(0), 1, 1)
        f_v = x.new_ones(x.size(0), 1, 1)

        if self.hue:
            f_h.uniform_(*self.hue)
        if self.saturation:
            f_s = f_s.uniform_(*self.saturation)
        if self.brightness:
            f_v = f_v.uniform_(*self.brightness)

        return RandomHSVFunction.apply(x, f_h, f_s, f_v)

    def transform(self, inputs):
        # Shuffle transform
        if np.random.rand() > 0.5:
            transforms = [self.adjust_contrast, self.adjust_hsv]
        else:
            transforms = [self.adjust_hsv, self.adjust_contrast]

        for t in transforms:
            inputs = t(inputs)

        return inputs

    def forward(self, inputs):
        _prob = inputs.new_full((inputs.size(0), ), self.prob)
        _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
        return inputs * (1 - _mask) + self.transform(inputs) * _mask


class RandomHSVFunction(Function):
    @staticmethod
    def forward(ctx, x, f_h, f_s, f_v):
        # ctx is a context object that can be used to stash information
        # for backward computation
        x = rgb2hsv(x)
        h = x[:, 0, :, :]
        h += (f_h * 255. / 360.)
        h = (h % 1)
        x[:, 0, :, :] = h
        x[:, 1, :, :] = x[:, 1, :, :] * f_s
        x[:, 2, :, :] = x[:, 2, :, :] * f_v
        x = torch.clamp(x, 0, 1)
        x = hsv2rgb(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
        return grad_input, None, None, None


class NormalizeLayer(nn.Module):
    """In order to certify radii in original coordinates rather than
    standardized coordinates, we add the Gaussian noise _before_ standardizing,
    which is why we have standardization be the first layer of the classifier
    rather than as a part of preprocessing as is typical."""
    def __init__(self):
        super(NormalizeLayer, self).__init__()

    def forward(self, inputs):
        return (inputs - 0.5) / 0.5
