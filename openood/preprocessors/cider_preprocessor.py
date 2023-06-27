import torchvision.transforms as tvs_trans

from openood.utils.config import Config

from .transform import Convert, interpolation_modes, normalization_dict


class CiderPreprocessor():
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

        if 'imagenet' in config.dataset.name:
            self.transform = tvs_trans.Compose([
                tvs_trans.RandomResizedCrop(size=self.image_size,
                                            scale=(0.4, 1.),
                                            interpolation=self.interpolation),
                tvs_trans.RandomHorizontalFlip(),
                tvs_trans.RandomApply(
                    [tvs_trans.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                tvs_trans.RandomGrayscale(p=0.2),
                tvs_trans.ToTensor(),
                tvs_trans.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            self.transform = tvs_trans.Compose([
                Convert('RGB'),
                tvs_trans.RandomResizedCrop(size=self.image_size,
                                            scale=(0.2, 1.),
                                            interpolation=self.interpolation),
                tvs_trans.RandomHorizontalFlip(),
                tvs_trans.RandomApply(
                    [tvs_trans.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                tvs_trans.RandomGrayscale(p=0.2),
                tvs_trans.ToTensor(),
                tvs_trans.Normalize(mean=self.mean, std=self.std),
            ])

        self.transform = TwoCropTransform(self.transform)

    def setup(self, **kwargs):
        pass

    def __call__(self, image):
        return self.transform(image)


class TwoCropTransform:
    """Create two crops of the same image."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
