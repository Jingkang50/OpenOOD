from openood.preprocessors.pixmix_preprocessor import PixMixPreprocessor
from openood.utils import Config

from .base_preprocessor import BasePreprocessor
from .cutpaste_preprocessor import CutPastePreprocessor
from .draem_preprocessor import DRAEMPreprocessor
from .test_preprocessor import TestStandardPreProcessor
from .pixmix_preprocessor import PixMixPreprocessor
from .patch_preprocessor import PatchStandardPreProcessor
from .patch_gts_preprocessor import PatchGTStandardPreProcessor

import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms as tvs_trans


normalization_dict = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'cifar100_openmax': [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]],
    'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    'svhn': [[0.431, 0.430, 0.446], [0.196, 0.198, 0.199]],
    'cifar10-3': [[-31.7975, -31.7975, -31.7975], [42.8907, 42.8907, 42.8907]],
    'covid': [[0.4907, 0.4907, 0.4907], [0.2697, 0.2697, 0.2697]],
}

center_crop_dict = {
    28: 28,
    32: 32,
    224: 256,
    256: 256,
    299: 320,
    331: 352,
    384: 384,
    480: 480
}

interpolation_modes = {
    'nearest': tvs_trans.InterpolationMode.NEAREST,
    'bilinear': tvs_trans.InterpolationMode.BILINEAR,
}

class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def get_preprocessor(split, config: Config):
  train_preprocessors = {
    'base': BasePreprocessor,
    'DRAEM': DRAEMPreprocessor,
    'cutpaste': CutPastePreprocessor,
    'pixmix': PixMixPreprocessor, 
  }
  if split == "train":
    return train_preprocessors[config.preprocessor.name](config)    # need to pass in config.preprocessor
  elif split == 'patch' or split == 'patchTest' or split == 'patchTestGood':
    return PatchStandardPreProcessor(config)
  elif split == 'patchGT':
    return  PatchGTStandardPreProcessor(config)
  else:    # for test and val
    return TestStandardPreProcessor[split](split, config)



"""Base augmentations operators."""

# ImageNet code should change this value
IMAGE_SIZE = 32

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


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
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
  a,b = get_ab(beta)
  img1, img2 = img1 * 2 - 1, img2 * 2 - 1
  out = a * img1 + b * img2
  return (out + 1) / 2

def multiply(img1, img2, beta):
  a,b = get_ab(beta)
  img1, img2 = img1 * 2, img2 * 2
  out = (img1 ** a) * (img2.clip(1e-37) ** b)
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