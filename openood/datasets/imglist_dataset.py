import ast
import io
import logging
import os

import torch
from PIL import Image, ImageFile

from openood.preprocessors import BasePreprocessor
from openood.preprocessors.transform import TestStandard, TrainStandard

from .base_dataset import BaseDataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImglistDataset(BaseDataset):
    def __init__(self,
                 name,
                 split,
                 interpolation,
                 image_size,
                 imglist_pth,
                 data_dir,
                 num_classes,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 preprocessor=None,
                 **kwargs):
        super(ImglistDataset, self).__init__(**kwargs)

        self.name = name
        self.image_size = image_size
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir

        if preprocessor is None:
            preprocessor = BasePreprocessor()
        self.preprocessor = preprocessor

        if split == 'train':
            self.transform_image = TrainStandard(name, image_size,
                                                 interpolation,
                                                 self.preprocessor)
        else:
            self.transform_image = TestStandard(name, image_size,
                                                interpolation,
                                                self.preprocessor)

        # some methods requires an auxiliary image without strong aug
        self.transform_aux_image = TestStandard(name, image_size,
                                                interpolation,
                                                self.preprocessor)

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
                if type(self.preprocessor).__name__ == 'DRAEMPreprocessor':
                    self.preprocessor.setup(path, self.name)

                image = Image.open(buff).convert('RGB')
                sample['data'] = self.transform_image(image)
        # sample['data_aux'] = self.transform_aux_image(image)
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
