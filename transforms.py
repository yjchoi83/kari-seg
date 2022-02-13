import torch
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

class ImageAug:
    def __init__(self, train):
        if train:
            self.aug = A.Compose([A.HorizontalFlip(p=0.5),
                                  A.VerticalFlip(p=0.5),
                                  #A.ShiftScaleRotate(p=0.5),
                                  A.RandomCrop(512, 512),
                                  ToTensorV2()])
        else:
            self.aug = A.Compose([#A.HorizontalFlip(p=0.5),
                                  #A.VerticalFlip(p=0.5),
                                  #A.ShiftScaleRotate(p=0.5),
                                  A.RandomCrop(512, 512),
                                  ToTensorV2()])
        
        # Cityscapes-specific
        self.void_idxes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_idxes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']
        self.ignore_idx = 0
        self.class_map = dict(zip(self.valid_idxes, range(1,20)))
        self.class_map[0] = self.ignore_idx
        
    def __call__(self, img, mask_img):
        transformed = self.aug(image=np.asarray(img), mask=np.asarray(mask_img))
        # remap mask labels
        trans_mask = transformed['mask']
        trans_mask = self.remap_idxes(trans_mask)
        return transformed['image']/255.0, trans_mask

    def remap_idxes(self, mask):
        mask = torch.where(mask >= 1000, mask.div(1000, rounding_mode='floor'), mask)
        for void_idx in self.void_idxes:
            mask[mask == void_idx] = self.ignore_idx
        for valid_idx in self.valid_idxes:
            mask[mask == valid_idx] = self.class_map[valid_idx]
        return mask

   
def get_transforms(train):
    transforms = ImageAug(train)
    return transforms