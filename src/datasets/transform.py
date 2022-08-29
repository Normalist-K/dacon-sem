import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']

def get_transform(resize):
    train_transform = A.Compose([
        A.Resize(96, 64, always_apply=True),
        # A.GaussianBlur(blur_limit=7, always_apply=True),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.05, 
        #                    scale_limit=0.05, 
        #                    rotate_limit=2, 
        #                    p=0.5),
        A.Normalize(mean=[0.5],std=[0.5], max_pixel_value=1.),
        ToTensorV2()
    ])

    infer_transform = A.Compose([
        A.Resize(96, 64, always_apply=True),
        # A.GaussianBlur(blur_limit=3, always_apply=True),
        A.Normalize(mean=[0.5],std=[0.5], max_pixel_value=1.),
        ToTensorV2()
    ])

    return (train_transform, infer_transform)