import random
import math
from torchvision import transforms
import torch 


class CPScarPreprocessor(object):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """
    def __init__(self, width=[2, 16], height=[10, 25], rotation=[-45, 45], **kwags):
        self.width = width
        self.height = height
        self.rotation = rotation
        
    def concat_transform(self, after_preprocessor_transform):
        self.after_preprocessor_transform = after_preprocessor_transform
        return self
    
    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]
        
        # cut region
        cut_w = random.uniform(*self.width)
        cut_h = random.uniform(*self.height)
        
        from_h = int(random.uniform(0, h - cut_h))
        from_w = int(random.uniform(0, w - cut_w))
        
        box = [from_w, from_h, from_w + cut_w, from_h + cut_h]
        patch = img.crop(box)
        
        if self.colorJitter:
            patch = self.colorJitter(patch)

        # rotate
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg, expand=True)
        
        # paste
        to_location_h = int(random.uniform(0, h - patch.size[0]))
        to_location_w = int(random.uniform(0, w - patch.size[1]))

        mask = patch.split()[-1]
        patch = patch.convert("RGB")
        
        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)
        
        if self.after_preprocessor_transform:
            img = self.after_preprocessor_transform(img)
            augmented = self.after_preprocessor_transform(augmented)
        
        return img, augmented