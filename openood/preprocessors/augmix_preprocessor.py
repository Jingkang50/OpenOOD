import torchvision.transforms as tvs_trans

from openood.utils.config import Config

from .transform import Convert, interpolation_modes, normalization_dict


class AugMixPreprocessor():
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

        self.severity = config.preprocessor.severity
        self.mixture_width = config.preprocessor.mixture_width
        self.alpha = config.preprocessor.alpha
        self.chain_depth = config.preprocessor.chain_depth
        self.all_ops = config.preprocessor.all_ops
        self.jsd = config.trainer.trainer_args.jsd

        self.augmix = tvs_trans.AugMix(severity=self.severity,
                                       mixture_width=self.mixture_width,
                                       chain_depth=self.chain_depth,
                                       alpha=self.alpha,
                                       all_ops=self.all_ops,
                                       interpolation=self.interpolation)
        self.normalize = tvs_trans.Compose([
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean=self.mean, std=self.std),
        ])

        if 'imagenet' in config.dataset.name:
            self.transform = tvs_trans.Compose([
                tvs_trans.RandomResizedCrop(self.image_size,
                                            interpolation=self.interpolation),
                tvs_trans.RandomHorizontalFlip(0.5),
            ])
        elif 'aircraft' in config.dataset.name or 'cub' in config.dataset.name:
            self.transform = tvs_trans.Compose([
                tvs_trans.Resize(self.pre_size,
                                 interpolation=self.interpolation),
                tvs_trans.RandomCrop(self.image_size),
                tvs_trans.RandomHorizontalFlip(),
            ])
        else:
            self.transform = tvs_trans.Compose([
                Convert('RGB'),
                tvs_trans.Resize(self.pre_size,
                                 interpolation=self.interpolation),
                tvs_trans.CenterCrop(self.image_size),
                tvs_trans.RandomHorizontalFlip(),
                tvs_trans.RandomCrop(self.image_size, padding=4),
            ])

    def setup(self, **kwargs):
        pass

    def __call__(self, image):
        if self.jsd:
            orig = self.transform(image)
            aug1 = self.normalize(self.augmix(orig))
            aug2 = self.normalize(self.augmix(orig))
            return self.normalize(orig), aug1, aug2
        else:
            return self.normalize(self.augmix(self.transform(image)))
