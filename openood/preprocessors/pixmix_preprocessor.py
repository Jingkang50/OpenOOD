import os

import numpy as np
import torch
import torchvision.transforms as tvs_trans
from PIL import Image as Image
from PIL import ImageEnhance, ImageOps

from .base_preprocessor import BasePreprocessor
from .transform import Convert, interpolation_modes, normalization_dict

resize_list = {
    'osr': 32,
    'mnist': 32,
    'cifar10': 36,
    'cifar100': 36,
    'tin': 72,
    'imagenet': 256,
    'imagenet200': 256,
    'aircraft': 512,
    'cub': 512,
}  # set mnist bymyself, imagenet was set to 224 by author, but 256 here


class PixMixPreprocessor(BasePreprocessor):
    def __init__(self, config):
        self.pre_size = config.dataset.pre_size
        self.dataset_name = config.dataset.name.split('_')[0]
        self.image_size = config.dataset.image_size
        self.interpolation = interpolation_modes[config.dataset.interpolation]
        normalization_type = config.dataset.normalization_type
        if normalization_type in normalization_dict.keys():
            self.mean = normalization_dict[normalization_type][0]
            self.std = normalization_dict[normalization_type][1]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        self.normalize = tvs_trans.Normalize(self.mean, self.std)
        self.tensorize = tvs_trans.ToTensor()

        self.args = config.preprocessor.preprocessor_args

        if 'imagenet' in config.dataset.name:
            self.transform = tvs_trans.Compose([
                tvs_trans.RandomResizedCrop(self.image_size,
                                            interpolation=self.interpolation),
                tvs_trans.RandomHorizontalFlip(0.5),
            ])
        elif 'aircraft' in config.dataset.name or 'cub' in config.dataset.name:
            self.transform = tvs_trans.Compose([
                tvs_trans.Resize(self.pre_size,
                                 interpolation=self.interpolation),
                tvs_trans.RandomCrop(self.image_size),
                tvs_trans.RandomHorizontalFlip(),
            ])
        else:
            self.transform = tvs_trans.Compose([
                Convert('RGB'),
                tvs_trans.Resize(self.pre_size,
                                 interpolation=self.interpolation),
                tvs_trans.CenterCrop(self.image_size),
                tvs_trans.RandomHorizontalFlip(),
                tvs_trans.RandomCrop(self.image_size, padding=4),
            ])

        self.mixing_set_transform = tvs_trans.Compose([
            tvs_trans.Resize(resize_list[self.dataset_name]),
            tvs_trans.RandomCrop(self.image_size)
        ])

        with open(self.args.mixing_set_dir, 'r') as f:
            self.mixing_list = f.readlines()

    def __call__(self, image):
        # ? need to add random seed ?
        rnd_idx = np.random.choice(len(self.mixing_list))
        mixing_pic_dir = self.mixing_list[rnd_idx].strip('\n')

        mixing_pic = Image.open(
            os.path.join('./data/images_classic/',
                         mixing_pic_dir)).convert('RGB')
        return self.pixmix(image, mixing_pic)

    def augment_input(self, image):
        aug_list = augmentations_all if self.args.all_ops else augmentations
        op = np.random.choice(aug_list)
        return op(image.copy(), self.args.aug_severity, self.image_size)

    def pixmix(self, orig, mixing_pic):
        mixings = [add, multiply]
        orig = self.transform(orig)

        # do basic augmentation first
        mixing_pic = self.mixing_set_transform(mixing_pic)

        if np.random.random() < 0.5:
            mixed = self.tensorize(self.augment_input(orig))
        else:
            mixed = self.tensorize(orig)

        for _ in range(np.random.randint(self.args.k + 1)):

            if np.random.random() < 0.5:
                aug_image_copy = self.tensorize(self.augment_input(orig))
            else:
                aug_image_copy = self.tensorize(mixing_pic)

            mixed_op = np.random.choice(mixings)

            mixed = mixed_op(mixed, aug_image_copy, self.args.beta)
            mixed = torch.clip(mixed, 0, 1)

        return self.normalize(mixed)


"""Base augmentations operators."""

#########################################################
#################### AUGMENTATIONS ######################
#########################################################


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
    Returns:
      An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.

    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
    Returns:
      A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _, IMAGE_SIZE):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _, IMAGE_SIZE):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level, IMAGE_SIZE):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level, IMAGE_SIZE):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level, IMAGE_SIZE):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level, IMAGE_SIZE):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, level, 0, 0, 1, 0),
                             resample=Image.BILINEAR)


def shear_y(pil_img, level, IMAGE_SIZE):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, 0, 0, level, 1, 0),
                             resample=Image.BILINEAR)


def translate_x(pil_img, level, IMAGE_SIZE):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, 0, level, 0, 1, 0),
                             resample=Image.BILINEAR)


def translate_y(pil_img, level, IMAGE_SIZE):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, 0, 0, 0, 1, level),
                             resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level, IMAGE_SIZE):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level, IMAGE_SIZE):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level, IMAGE_SIZE):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level, IMAGE_SIZE):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]

#########################################################
######################## MIXINGS ########################
#########################################################


def get_ab(beta):
    if np.random.random() < 0.5:
        a = np.float32(np.random.beta(beta, 1))
        b = np.float32(np.random.beta(1, beta))
    else:
        a = 1 + np.float32(np.random.beta(1, beta))
        b = -np.float32(np.random.beta(1, beta))
    return a, b


def add(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2 - 1, img2 * 2 - 1
    out = a * img1 + b * img2
    return (out + 1) / 2


def multiply(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2, img2 * 2
    out = (img1**a) * (img2.clip(1e-37)**b)
    return out / 2


########################################
##### EXTRA MIXIMGS (EXPREIMENTAL) #####
########################################


def invert(img):
    return 1 - img


def screen(img1, img2, beta):
    img1, img2 = invert(img1), invert(img2)
    out = multiply(img1, img2, beta)
    return invert(out)


def overlay(img1, img2, beta):
    case1 = multiply(img1, img2, beta)
    case2 = screen(img1, img2, beta)
    if np.random.random() < 0.5:
        cond = img1 < 0.5
    else:
        cond = img1 > 0.5
    return torch.where(cond, case1, case2)


def darken_or_lighten(img1, img2, beta):
    if np.random.random() < 0.5:
        cond = img1 < img2
    else:
        cond = img1 > img2
    return torch.where(cond, img1, img2)


def swap_channel(img1, img2, beta):
    channel = np.random.randint(3)
    img1[channel] = img2[channel]
    return img1
