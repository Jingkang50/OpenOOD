from .utils import center_crop_dict, normalization_dict, interpolation_modes, Convert
import torchvision.transforms as tvs_trans
from openood.utils.config import Config
from PIL import Image


class PatchGTStandardPreProcessor:
    def __init__(self,
                 config: Config):
        image_size = config.dataset.image_size
        
        self.transform = tvs_trans.Compose([
            tvs_trans.Resize((image_size, image_size)),
            tvs_trans.CenterCrop(224),
            tvs_trans.ToTensor()
        ])

    def __call__(self, image):
        return self.transform(image)