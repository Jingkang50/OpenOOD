from .transform_statics import center_crop_dict, normalization_dict, interpolation_modes, Convert
import torchvision.transforms as tvs_trans
from openood.utils.config import Config
class BasePreprocessor():
    """
    For train dataset standard transformation.
    """

    def __init__(self,
                 config: Config
                 ):
        dataset_name = config.dataset.name.split('_')[0]
        image_size = config.dataset.image_size
        pre_size = center_crop_dict[image_size]
        if dataset_name in normalization_dict.keys():
            mean = normalization_dict[dataset_name][0]
            std = normalization_dict[dataset_name][1]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        interpolation = interpolation_modes[config.dataset["train"].interpolation]

        self.transform = tvs_trans.Compose([
                Convert('RGB'),
                tvs_trans.Resize(pre_size, interpolation=interpolation),
                tvs_trans.CenterCrop(image_size),
                tvs_trans.RandomHorizontalFlip(),
                tvs_trans.RandomCrop(image_size, padding=4),
                tvs_trans.ToTensor(),
                tvs_trans.Normalize(mean=mean, std=std),
            ])

    def __call__(self, image):
        return self.transform(image)
