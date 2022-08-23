import cv2
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class SEMDataset(Dataset):
    def __init__(
        self, 
        sem_path_list, 
        depth_path_list=None, 
        transform=None, 
    ):
    
        self.sem_path_list = sem_path_list
        self.depth_path_list = depth_path_list
        self.transform = transform
        
    def __getitem__(self, idx):
        sem_path = self.sem_path_list[idx]
        sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE) / 255.

        if self.depth_path_list is not None:
            depth_path = self.depth_path_list[idx]
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE) / 255.

            if self.transform:
                transformed = self.transform(image=sem_img, mask=depth_img)
                sem_img = transformed['image']
                depth_img = transformed['mask'].unsqueeze(dim=0)
            else:
                transform = ToTensorV2()
                sem_img = transform(image=sem_img)['image']
                depth_img = transform(image=depth_img)['image']

            return (sem_path, depth_path), sem_img.float(), depth_img.float()

        else:
            if self.transform:
                transformed = self.transform(image=sem_img)
                sem_img = transformed['image']
            else:
                transform = ToTensorV2()
                sem_img = transform(image=sem_img)['image']

            return sem_path, sem_img.float()


    def __len__(self):
        return len(self.sem_path_list)
