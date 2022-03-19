import random
import math
import numpy as np
from torchvision import transforms
import torch 
    
    
class CutOutPreprocessor(object):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """
    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwags):
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
      
    def concat_transform(self, after_preprocessor_transform):
        self.after_preprocessor_transform = after_preprocessor_transform
        return self
      
    def __call__(self, img):
        
        h = img.size[0]
        w = img.size[1]
        
        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h
        
        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1/self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()
        
        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))
        patch = np.zeros([cut_w, cut_h, 3], np.uint8)
        
        if self.colorJitter:
            patch = self.colorJitter(patch)
        
        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))
        
        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        augmented = img.copy()
        augmented.paste(patch, insert_box)
        
        if self.after_preprocessor_transform:
            img = self.after_preprocessor_transform(img)
            augmented = self.after_preprocessor_transform(augmented)
        
        return img, augmented