import ast
import io
import logging
import os

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Sampler

from .base_dataset import BaseDataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class ImglistExtraDataDataset(BaseDataset):
    def __init__(self,
                 name,
                 imglist_pth,
                 data_dir,
                 num_classes,
                 preprocessor,
                 data_aux_preprocessor,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 extra_data_pth=None,
                 extra_label_pth=None,
                 extra_percent=100,
                 **kwargs):
        super(ImglistExtraDataDataset, self).__init__(**kwargs)

        self.name = name
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        if dummy_read and dummy_size is None:
            raise ValueError(
                'if dummy_read is True, should provide dummy_size')

        self.orig_ids = list(range(len(self.imglist)))

        assert extra_data_pth is not None
        assert extra_label_pth is not None
        extra_data = np.load(extra_data_pth)
        extra_labels = np.load(extra_label_pth)
        assert len(extra_data) == len(extra_labels)

        self.extra_num = int(len(extra_labels) * extra_percent / 100.)
        self.total_num = len(self.imglist) + self.extra_num

        rng = np.random.RandomState(0)
        indices = rng.permutation(len(extra_labels))
        self.extra_data = extra_data[indices[:self.extra_num]]
        self.extra_labels = extra_labels[indices[:self.extra_num]]
        self.extra_ids = list(
            set(range(self.total_num)) - set(range(len(self.imglist))))

    def __len__(self):
        return self.total_num

    def getitem(self, index):
        if index in self.orig_ids:
            line = self.imglist[index].strip('\n')
            tokens = line.split(' ', 1)
            image_name, extra_str = tokens[0], tokens[1]
            if self.data_dir != '' and image_name.startswith('/'):
                raise RuntimeError('image_name starts with "/"')
            path = os.path.join(self.data_dir, image_name)
            sample = dict()
            sample['image_name'] = image_name
            kwargs = {'name': self.name, 'path': path, 'tokens': tokens}
            # some preprocessor methods require setup
            self.preprocessor.setup(**kwargs)
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
                    # if you use dic the code below will need ['label']
                    sample['label'] = 0
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
        else:
            ind = index - len(self.imglist)
            image = Image.fromarray(self.extra_data[ind])

            sample = dict()
            sample['image_name'] = str(ind)  # dummy name
            sample['data'] = self.transform_image(image)
            sample['data_aux'] = self.transform_aux_image(image)
            sample['label'] = self.extra_labels[ind]

            # Generate Soft Label
            soft_label = torch.Tensor(self.num_classes)
            if sample['label'] < 0:
                soft_label.fill_(1.0 / self.num_classes)
            else:
                soft_label.fill_(0)
                soft_label[sample['label']] = 1
            sample['soft_label'] = soft_label

            return sample


class TwoSourceSampler(Sampler):
    def __init__(self, real_inds, syn_inds, batch_size, real_ratio=0.5):
        assert len(real_inds) == 50000
        self.real_inds = real_inds
        self.syn_inds = syn_inds
        self.batch_size = batch_size
        self.real_batch_size = int(self.batch_size * real_ratio)
        self.syn_batch_size = self.batch_size - self.real_batch_size
        if real_ratio == 0:
            assert self.real_batch_size == 0
        elif real_ratio == 1:
            assert self.syn_batch_size == 0

        self.num_batches = int(np.ceil(len(self.real_inds) / self.batch_size))
        super().__init__(None)

    def __iter__(self):
        batch_counter = 0
        real_inds_shuffled = [
            self.real_inds[i] for i in torch.randperm(len(self.real_inds))
        ]
        syn_inds_shuffled = [
            self.syn_inds[i] for i in torch.randperm(len(self.syn_inds))
        ]

        real_offset = 0
        syn_offset = 0
        while batch_counter < self.num_batches:
            real_batch = real_inds_shuffled[
                real_offset:min(real_offset +
                                self.real_batch_size, len(real_inds_shuffled))]
            real_offset += self.real_batch_size

            syn_batch = syn_inds_shuffled[
                syn_offset:min(syn_offset +
                               self.syn_batch_size, len(syn_inds_shuffled))]
            syn_offset += self.syn_batch_size

            batch = real_batch + syn_batch
            np.random.shuffle(batch)
            yield batch
            batch_counter += 1

    def __len__(self):
        return self.num_batches
