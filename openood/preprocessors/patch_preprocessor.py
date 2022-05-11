from .transform_statics import normalization_dict
import torchvision.transforms as tvs_trans
from openood.utils.config import Config
from PIL import Image

class PatchStandardPreProcessor:
    def __init__(self,
                 config: Config):
        dataset_name = config.dataset.name.split('_')[0]
        image_size = config.dataset.image_size
        if dataset_name in normalization_dict.keys():
            mean = normalization_dict[dataset_name][0]
            std = normalization_dict[dataset_name][1]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        self.transform = tvs_trans.Compose([
            tvs_trans.Resize((image_size, image_size), Image.ANTIALIAS),
            tvs_trans.CenterCrop(224),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean, std),
        ])

    def __call__(self, image):
        return self.transform(image)