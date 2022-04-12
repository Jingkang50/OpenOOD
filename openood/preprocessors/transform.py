import numpy as np
import torch
import torchvision.transforms as tvs_trans

normalization_dict = {
    'mnist': [[0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]],
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    'cifar10-3': [[-31.7975, -31.7975, -31.7975], [42.8907, 42.8907, 42.8907]],
}

center_crop_dict = {28: 28, 32: 32, 224: 256, 299: 320, 331: 352}

interpolation_modes = {
    'nearest': tvs_trans.InterpolationMode.NEAREST,
    'bilinear': tvs_trans.InterpolationMode.BILINEAR,
}


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class TrainStandard:
    def __init__(self,
                 name: str,
                 image_size: int,
                 interpolation: str = 'bilinear',
                 CustomPreprocessor=None):
        pre_size = center_crop_dict[image_size]
        dataset_name = name.split('_')[0]
        if dataset_name in normalization_dict.keys():
            mean = normalization_dict[dataset_name][0]
            std = normalization_dict[dataset_name][1]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        interpolation = interpolation_modes[interpolation]

        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(pre_size, interpolation=interpolation),
            tvs_trans.CenterCrop(image_size),
            tvs_trans.RandomHorizontalFlip(),
            tvs_trans.RandomCrop(image_size, padding=4),
            CustomPreprocessor,
            tvs_trans.ToTensor(),
            tvs_trans.Lambda(
                lambda x: global_contrast_normalization(x, scale='l1')),
            tvs_trans.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class TestStandard:
    def __init__(self,
                 name: str,
                 image_size: int,
                 interpolation: str = 'bilinear',
                 CustomPreprocessor=None):

        pre_size = center_crop_dict[image_size]
        dataset_name = name.split('_')[0]
        if dataset_name in normalization_dict.keys():
            mean = normalization_dict[dataset_name][0]
            std = normalization_dict[dataset_name][1]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        interpolation = interpolation_modes[interpolation]

        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(pre_size, interpolation=interpolation),
            tvs_trans.CenterCrop(image_size),
            CustomPreprocessor,
            tvs_trans.ToTensor(),
            tvs_trans.Lambda(
                lambda x: global_contrast_normalization(x, scale='l1')),
            tvs_trans.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """Apply global contrast normalization to tensor, i.e. subtract mean across
    features (pixels) and normalize by scale, which is either the standard
    deviation, L1- or L2-norm across features (pixels).

    Note this is a *per sample* normalization globally across features (and not
    across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x**2)) / n_features

    x /= x_scale

    return x
