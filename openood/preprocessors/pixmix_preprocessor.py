from re import I, S
import PIL.Image as Image
import torch
import numpy as np
import torchvision.transforms as tfm
from .utils import *

resize_list = {'mnist': 32, 'cifar10': 36, 'imagenet': 256}  # set mnist bymyself, imagenet was set to 224 by author, but 256 here

class PixMixPreprocessor:
    def __init__(self, config):
        self.config = config
        dataset_name = config.dataset.name.split('_')[0]
        image_size = config.dataset.image_size
        self.args = self.config.preprocessor.preprocessor_args
        self.tensorize = tfm.ToTensor()
        if dataset_name in normalization_dict.keys():
            mean = normalization_dict[dataset_name][0]
            std = normalization_dict[dataset_name][1]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        self.normalize = tfm.Normalize(mean, std)    # ? use which one ?
        pre_size = center_crop_dict[image_size]
        interpolation = interpolation_modes[config.dataset["train"].interpolation]

        self.transform = tvs_trans.Compose([
                Convert('RGB'),
                tvs_trans.Resize(pre_size, interpolation=interpolation),
                tvs_trans.CenterCrop(image_size),
                tvs_trans.RandomHorizontalFlip(),
                tvs_trans.RandomCrop(image_size, padding=4),
                self.tensorize,
                self.normalize,
            ])

        self.mixing_set_transform = tfm.Compose([tfm.Resize(resize_list[dataset_name]), tfm.RandomCrop(self.image_size)])

        with open(self.args.mixing_set_dir, 'r') as f:
            self.mixing_list = f.readlines()

    def __call__(self, image):
        # ? need to add random seed ? 
        rnd_idx = np.random.choice(len(self.mixing_list))
        mixing_pic_dir = self.mixing_list[rnd_idx].strip("\n")
        mixing_pic = Image.open(mixing_pic_dir).convert('RGB')
        return self.pixmix(image, mixing_pic)


    def augment_input(self, image):
        aug_list = augmentations_all if self.args.all_ops else augmentations
        op = np.random.choice(aug_list)
        return op(image.copy(), self.args.aug_severity)

    def pixmix(self, orig, mixing_pic):
        mixings = [add, multiply]
        orig = self.transform(orig)   # do basic augmentation first
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

    