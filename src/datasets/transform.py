import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']

def get_transform(resize):
    simulation_transform = A.Compose([
        A.Resize(resize[0], resize[1], always_apply=True),
        ToTensorV2()
    ])

    infer_transform = A.Compose([
        A.Resize(resize[0], resize[1], always_apply=True),
        ToTensorV2()
    ])

    train_transform = A.Compose([
        A.Resize(resize[0], resize[1], always_apply=True),
        ToTensorV2()
    ])

    return (simulation_transform, infer_transform, train_transform)

# def get_transform(resize):
#     simulation_transform = A.Compose([
#         A.Resize(resize[0], resize[1], always_apply=True),
#         A.Normalize(mean=[0.3920],std=[0.2262], max_pixel_value=1.),
#         ToTensorV2()
#     ])

#     infer_transform = A.Compose([
#         A.Resize(resize[0], resize[1], always_apply=True),
#         A.Normalize(mean=[0.3920],std=[0.2262], max_pixel_value=1.),
#         ToTensorV2()
#     ])

#     return (simulation_transform, infer_transform)

def get_transform1(resize):
    simulation_transform = A.Compose([
        A.Resize(resize[0], resize[1], always_apply=True),
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
        A.Resize(resize[0], resize[1], always_apply=True),
        # A.GaussianBlur(blur_limit=3, always_apply=True),
        # A.Normalize(mean=[0.4532],std=[0.2577], max_pixel_value=1.),
        A.Normalize(mean=[0.5],std=[0.5], max_pixel_value=1.),
        ToTensorV2()
    ])

    return (simulation_transform, infer_transform)

def get_transform2(resize):
    simulation_transform = A.Compose([
        A.Resize(resize[0], resize[1], always_apply=True),
        A.GaussianBlur(blur_limit=(3, 11), p=0.5),
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
        A.Resize(resize[0], resize[1], always_apply=True),
        # A.GaussianBlur(blur_limit=3, always_apply=True),
        # A.Normalize(mean=[0.4532],std=[0.2577], max_pixel_value=1.),
        A.Normalize(mean=[0.5],std=[0.5], max_pixel_value=1.),
        ToTensorV2()
    ])

    return (simulation_transform, infer_transform)
    