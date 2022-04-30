import math
import random
import torch
import torchvision.transforms as tvs_trans


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class CutPastePreprocessor(object):
    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwags):
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self.before_preprocessor_transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(256, interpolation=tvs_trans.InterpolationMode.BILINEAR),
            tvs_trans.CenterCrop(256),
            tvs_trans.RandomHorizontalFlip(),
            tvs_trans.RandomCrop(256, padding=4),
        ])
        self.after_preprocessor_transform = tvs_trans.Compose([
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        img = self.before_preprocessor_transform(img)

        h = img.size[0]
        w = img.size[1]

        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(0.02, 0.15) * w * h

        # sample in log space
        log_ratio = torch.log(
            torch.tensor((self.aspect_ratio, 1 / self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [
            from_location_w, from_location_h, from_location_w + cut_w,
            from_location_h + cut_h
        ]
        patch = img.crop(box)

        # if self.colorJitter:
        #     patch = self.colorJitter(patch)

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))

        insert_box = [
            to_location_w, to_location_h, to_location_w + cut_w,
            to_location_h + cut_h
        ]
        augmented = img.copy()
        augmented.paste(patch, insert_box)

        img = self.after_preprocessor_transform(img)
        augmented = self.after_preprocessor_transform(augmented)

        return img, augmented
