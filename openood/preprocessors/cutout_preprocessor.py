import numpy as np
import torch
import torchvision.transforms as tvs_trans

from openood.utils.config import Config
from .transform import Convert, interpolation_modes, normalization_dict


class CutoutPreprocessor():
    def __init__(self, config: Config):
        self.pre_size = config.dataset.pre_size
        self.image_size = config.dataset.image_size
        self.interpolation = interpolation_modes[config.dataset.interpolation]
        normalization_type = config.dataset.normalization_type
        if normalization_type in normalization_dict.keys():
            self.mean = normalization_dict[normalization_type][0]
            self.std = normalization_dict[normalization_type][1]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]

        self.n_holes = config.preprocessor.n_holes
        self.length = config.preprocessor.length

        if 'imagenet' in config.dataset.name:
            self.transform = tvs_trans.Compose([
                tvs_trans.RandomResizedCrop(self.image_size,
                                            interpolation=self.interpolation),
                tvs_trans.RandomHorizontalFlip(0.5),
                tvs_trans.ToTensor(),
                tvs_trans.Normalize(mean=self.mean, std=self.std),
                Cutout(n_holes=self.n_holes, length=self.length)
            ])
        elif 'aircraft' in config.dataset.name or 'cub' in config.dataset.name:
            self.transform = tvs_trans.Compose([
                tvs_trans.Resize(self.pre_size,
                                 interpolation=self.interpolation),
                tvs_trans.RandomCrop(self.image_size),
                tvs_trans.RandomHorizontalFlip(),
                tvs_trans.ColorJitter(brightness=32. / 255., saturation=0.5),
                tvs_trans.ToTensor(),
                tvs_trans.Normalize(mean=self.mean, std=self.std),
                Cutout(n_holes=self.n_holes, length=self.length)
            ])
        else:
            self.transform = tvs_trans.Compose([
                Convert('RGB'),
                tvs_trans.Resize(self.pre_size,
                                 interpolation=self.interpolation),
                tvs_trans.CenterCrop(self.image_size),
                tvs_trans.RandomHorizontalFlip(),
                tvs_trans.RandomCrop(self.image_size, padding=4),
                tvs_trans.ToTensor(),
                tvs_trans.Normalize(mean=self.mean, std=self.std),
                Cutout(n_holes=self.n_holes, length=self.length)
            ])

    def setup(self, **kwargs):
        pass

    def __call__(self, image):
        return self.transform(image)


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length
            cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
