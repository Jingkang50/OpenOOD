import torchvision.transforms as tvs_trans
from PIL import Image

from openood.utils.config import Config


class PatchGTStandardPreProcessor:
    def __init__(self, config: Config):
        image_size = config.dataset.image_size

        self.transform = tvs_trans.Compose([
            tvs_trans.Resize((image_size, image_size)),
            tvs_trans.CenterCrop(224),
            tvs_trans.ToTensor()
        ])

    def __call__(self, image):
        return self.transform(image)
