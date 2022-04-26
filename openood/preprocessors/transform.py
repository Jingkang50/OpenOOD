import torchvision.transforms as tvs_trans

normalization_dict = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    'covid': [[0.4907, 0.4907, 0.4907], [0.2697, 0.2697, 0.2697]],
}

center_crop_dict = {28: 28, 32: 32, 224: 224, 299: 320, 331: 352, 480: 480}

interpolation_modes = {
    'nearest': 1,
    'bilinear': 2,
}


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class TrainStandard:
    def __init__(self,
                 name: str,
                 image_size: int,
                 interpolation: str = 'bilinear',
                 CustomPreprocessor=None):
        pre_size = center_crop_dict[image_size]
        dataset_name = name.split('_')[0]
        if dataset_name in normalization_dict.keys():
            mean = normalization_dict[dataset_name][0]
            std = normalization_dict[dataset_name][1]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        interpolation = interpolation_modes[interpolation]

        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(pre_size, interpolation=interpolation),
            tvs_trans.CenterCrop(image_size),
            tvs_trans.RandomHorizontalFlip(),
            tvs_trans.RandomCrop(image_size, padding=4),
            CustomPreprocessor,
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class TestStandard:
    def __init__(self,
                 name: str,
                 image_size: int,
                 interpolation: str = 'bilinear',
                 CustomPreprocessor=None):

        pre_size = center_crop_dict[image_size]
        dataset_name = name.split('_')[0]
        if dataset_name in normalization_dict.keys():
            mean = normalization_dict[dataset_name][0]
            std = normalization_dict[dataset_name][1]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        interpolation = interpolation_modes[interpolation]

        self.transform = tvs_trans.Compose([
        tvs_trans.Resize((image_size, image_size)),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def __call__(self, image):
        return self.transform(image)
