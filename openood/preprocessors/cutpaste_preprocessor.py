import math
import random

import torch
import torchvision.transforms as tvs_trans

from .base_preprocessor import BasePreprocessor
from .transform import Convert, normalization_dict


class CutPastePreprocessor(BasePreprocessor):
    def __init__(
            self, config,
            split):  # modify, preprocessors unify to only passing in "config"
        self.args = config.preprocessor.preprocessor_args
        self.area_ratio = self.args.area_ratio
        self.aspect_ratio = self.args.aspect_ratio

        dataset_name = config.dataset.name.split('_')[0]
        image_size = config.dataset.image_size
        pre_size = config.dataset.pre_size
        if dataset_name in normalization_dict.keys():
            mean = normalization_dict[dataset_name][0]
            std = normalization_dict[dataset_name][1]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        self.before_preprocessor_transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(
                pre_size, interpolation=tvs_trans.InterpolationMode.BILINEAR),
            tvs_trans.CenterCrop(image_size),
            tvs_trans.RandomHorizontalFlip(),
            tvs_trans.RandomCrop(image_size, padding=4),
        ])
        self.after_preprocessor_transform = tvs_trans.Compose([
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        img = self.before_preprocessor_transform(img)

        h = img.size[0]
        w = img.size[1]

        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(0.02, 0.15) * w * h

        # sample in log space
        log_ratio = torch.log(
            torch.tensor((self.aspect_ratio, 1 / self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [
            from_location_w, from_location_h, from_location_w + cut_w,
            from_location_h + cut_h
        ]
        patch = img.crop(box)

        # if self.colorJitter:
        #     patch = self.colorJitter(patch)

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))

        insert_box = [
            to_location_w, to_location_h, to_location_w + cut_w,
            to_location_h + cut_h
        ]
        augmented = img.copy()
        augmented.paste(patch, insert_box)

        img = self.after_preprocessor_transform(img)
        augmented = self.after_preprocessor_transform(augmented)

        return img, augmented
