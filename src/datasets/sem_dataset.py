import cv2
from torch.utils.data import Dataset


class SEMDataset(Dataset):
    def __init__(
        self, 
        sem_path_list, 
        depth_path_list=None, 
        transform=None, 
        label_transform=None
    ):
    
        self.sem_path_list = sem_path_list
        self.depth_path_list = depth_path_list
        self.transform = transform
        self.label_transform = label_transform
        
    def __getitem__(self, idx):
        sem_path = self.sem_path_list[idx]
        sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            sem_img = self.transform(sem_img)

        if self.depth_path_list is not None:
            depth_path = self.depth_path_list[idx]
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            depth_img = depth_img / 255.

            if self.label_transform:
                depth_img = self.label_transform(depth_img)
                depth_img = depth_img.float()

            return (sem_path, depth_path), sem_img, depth_img
        
        else:
            return sem_path, sem_img
    
    def __len__(self):
        return len(self.sem_path_list)
