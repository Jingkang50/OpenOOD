import torchvision.transforms as tvs_trans


normalization_dict = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'cifar100_openmax': [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]],
    'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    'svhn': [[0.431, 0.430, 0.446], [0.196, 0.198, 0.199]],
    'cifar10-3': [[-31.7975, -31.7975, -31.7975], [42.8907, 42.8907, 42.8907]],
    'covid': [[0.4907, 0.4907, 0.4907], [0.2697, 0.2697, 0.2697]],
}

center_crop_dict = {
    28: 28,
    32: 32,
    224: 256,
    256: 256,
    299: 320,
    331: 352,
    384: 384,
    480: 480
}

interpolation_modes = {
    'nearest': tvs_trans.InterpolationMode.NEAREST,
    'bilinear': tvs_trans.InterpolationMode.BILINEAR,
}

class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)

