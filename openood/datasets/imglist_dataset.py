import ast
import io
import logging
import os
from typing import List

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
    split: str,
    interpolation: str = 'bilinear',
    image_size: int = 256,
    crop_size: int = 224,
):
    interpolation_modes = {
        'nearest': InterpolationMode.NEAREST,
        'bilinear': InterpolationMode.BILINEAR,
    }
    color_mode = 'RGB'

    interpolation = interpolation_modes[interpolation]

    if split == 'train':
        return trn.Compose([
            Convert(color_mode),
            trn.Resize(image_size, interpolation=interpolation),
            trn.CenterCrop(crop_size),
            trn.RandomHorizontalFlip(),
            trn.RandomCrop(image_size, padding=4),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])
    
    elif split == 'patch':
        return trn.Compose([
            # Convert(color_mode),
            trn.Resize(image_size, interpolation=interpolation),
            trn.CenterCrop(crop_size),
            # trn.RandomHorizontalFlip(),
            # trn.RandomCrop(image_size, padding=4),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])

    else:
        return trn.Compose([
            Convert(color_mode),
            trn.Resize(image_size, interpolation=interpolation),
            trn.CenterCrop(crop_size),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])


class ImglistDataset(BaseDataset):
    def __init__(self,
                 name,
                 split,
                 interpolation,
                 image_size,
                 imglist_pth,
                 data_dir,
                 num_classes,
                 preprocessor,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 crop_size = 224,
                 mean = [0.5, 0.5, 0.5],
                 std = [0.5 ,0.5, 0.5],
                 **kwargs):
        super(ImglistDataset, self).__init__(**kwargs)
        
        self.name = name
        self.image_size = image_size
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        # # TODO: mean and std are different from original value
        # mean, std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        self.transform_image = get_transforms(mean, std, split, interpolation,
                                              image_size,crop_size)
        self.transform_aux_image = get_transforms(mean, std, 'val',
                                                  interpolation, image_size,crop_size)
        self.num_classes = num_classes
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        if dummy_read and dummy_size is None:
            raise ValueError(
                'if dummy_read is True, should provide dummy_size')

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        path = os.path.join(self.data_dir, image_name)
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
                sample['data_aux'] = self.transform_aux_image(image)
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

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample
