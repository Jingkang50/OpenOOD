import ast
import io
import logging
import os
from typing import List

import numpy as np
import torch
import torchvision.transforms as trn
from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode

from .base_dataset import BaseDataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def get_transforms(
    mean: List[float],
    std: List[float],
    stage: str,
    interpolation: str = 'bilinear',
    image_size: int = 32,
):
    interpolation_modes = {
        'nearest': InterpolationMode.NEAREST,
        'bilinear': InterpolationMode.BILINEAR,
    }
    color_mode = 'RGB'

    interpolation = interpolation_modes[interpolation]

    if stage == 'train':
        return trn.Compose([
            Convert(color_mode),
            trn.Resize(image_size, interpolation=interpolation),
            trn.CenterCrop(image_size),
            trn.RandomHorizontalFlip(),
            trn.RandomCrop(image_size, padding=4),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])

    elif stage in ['val', 'test']:
        return trn.Compose([
            Convert(color_mode),
            trn.Resize(image_size, interpolation=interpolation),
            trn.CenterCrop(image_size),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])


class ImagenameDataset(BaseDataset):
    def __init__(self,
                 name,
                 stage,
                 interpolation,
                 image_size,
                 imglist,
                 root,
                 num_classes=10,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 **kwargs):
        super(ImagenameDataset, self).__init__(**kwargs)

        self.name = name
        self.image_size = image_size
        with open(imglist) as imgfile:
            self.imglist = imgfile.readlines()
        self.root = root
        mean, std = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        self.transform_image = get_transforms(mean, std, stage, interpolation,
                                              image_size)
        # basic image transformation for online clustering
        # (without augmentations)
        self.transform_aux_image = get_transforms(mean, std, 'test',
                                                  interpolation, image_size)

        self.num_classes = num_classes
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        if dummy_read and dummy_size is None:
            raise ValueError(
                'if dummy_read is True, should provide dummy_size')

        self.cluster_id = np.zeros(len(self.imglist), dtype=int)
        self.cluster_reweight = np.ones(len(self.imglist), dtype=float)

        # use pseudo labels for unlabeled dataset during training
        self.pseudo_label = np.array(-1 * np.ones(len(self.imglist)),
                                     dtype=int)
        self.ood_conf = np.ones(len(self.imglist), dtype=float)

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.root != '' and image_name.startswith('/'):
            raise RuntimeError('root not empty but image_name starts with "/"')
        path = os.path.join(self.root, image_name)
        sample = dict()
        sample['image_name'] = image_name
        try:
            if not self.dummy_read:
                with open(path, 'rb') as f:
                    content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
            if self.dummy_size is not None:
                sample['data'] = torch.rand(self.dummy_size)
            else:
                image = Image.open(buff).convert('RGB')
                sample['data'] = self.transform_image(image)
                sample['plain_data'] = self.transform_aux_image(image)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
            except AttributeError:
                sample['label'] = int(extra_str)
            # Generate Soft Label
            soft_label = torch.Tensor(self.num_classes)
            if sample['label'] < 0:
                soft_label.fill_(1.0 / self.num_classes)
            else:
                soft_label.fill_(0)
                soft_label[sample['label']] = 1
            sample['soft_label'] = soft_label
            # Deep Clustering Aux Label Assignment for
            # both labeled/unlabeled data
            sample['cluster_id'] = self.cluster_id[index]
            sample['cluster_reweight'] = self.cluster_reweight[index]

            # Deep Clustering Pseudo Label Assignment for unlabeled data
            sample['pseudo_label'] = self.pseudo_label[index]
            soft_pseudo_label = torch.Tensor(len(sample['soft_label']))
            if sample['pseudo_label'] == -1:
                soft_pseudo_label.fill_(1.0 / len(sample['soft_label']))
            else:
                soft_pseudo_label.fill_(0.0)
                soft_pseudo_label[sample['pseudo_label']] = 1.0
            sample['pseudo_softlabel'] = soft_pseudo_label
            sample['ood_conf'] = self.ood_conf[index]

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample
