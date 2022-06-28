import ast
import io
import logging
import os

import numpy as np
import torch
from PIL import Image, ImageFile

from .imglist_dataset import ImglistDataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class UDGDataset(ImglistDataset):
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
                 **kwargs):
        super(UDGDataset,
              self).__init__(name, imglist_pth, data_dir, num_classes,
                             preprocessor, data_aux_preprocessor, maxlen,
                             dummy_read, dummy_size, **kwargs)

        self.cluster_id = np.zeros(len(self.imglist), dtype=int)
        self.cluster_reweight = np.ones(len(self.imglist), dtype=float)

        # use pseudo labels for unlabeled dataset during training
        self.pseudo_label = np.array(-1 * np.ones(len(self.imglist)),
                                     dtype=int)
        self.ood_conf = np.ones(len(self.imglist), dtype=float)

    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('root not empty but image_name starts with "/"')
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
