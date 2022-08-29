import cv2
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class SEMDataset(Dataset):
    def __init__(
        self, 
        sem_path_list, 
        depth_path_list=None, 
        transform=None, 
        aux=False,
    ):
    
        self.sem_path_list = sem_path_list
        self.depth_path_list = depth_path_list
        if transform is not None:
            self.train_transform = transform[0]
            self.infer_transform = transform[1]
        else:
            self.train_transform = None
            self.infer_transform = None
        self.aux = aux

    def embedding(self, case):

        assert case in ['Case_1', 'Case_2', 'Case_3', 'Case_4']

        emb_dict = {
            'Case_1': 0,
            'Case_2': 1,
            'Case_3': 2,
            'Case_4': 3
        }
        return emb_dict[case]
        
    def __getitem__(self, idx):
        sem_path = self.sem_path_list[idx]
        sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE) / 255.

        case = sem_path.split('/')[-3]
        case_emb = self.embedding(case) if self.aux else -1

        if self.depth_path_list is not None:
            depth_path = self.depth_path_list[idx]
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE) / 255.

            if self.train_transform:
                transformed = self.train_transform(image=sem_img, mask=depth_img)
                sem_img = transformed['image']
                depth_img = transformed['mask'].unsqueeze(dim=0)
            else:
                transform = ToTensorV2()
                sem_img = transform(image=sem_img)['image']
                depth_img = transform(image=depth_img)['image']

            return (sem_path, depth_path), sem_img.float(), depth_img.float(), case_emb

        else:
            if self.infer_transform:
                transformed = self.infer_transform(image=sem_img)
                sem_img = transformed['image']
            else:
                transform = ToTensorV2()
                sem_img = transform(image=sem_img)['image']

            return sem_path, sem_img.float()


    def __len__(self):
        return len(self.sem_path_list)
