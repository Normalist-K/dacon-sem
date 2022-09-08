import random
import cv2
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class SEMDataset(Dataset):
    def __init__(
        self,
        sem_path_list=None,
        depth_path_list=None,
        train_sem_path_list=None,
        transform=None, 
        aux=False,
    ):
    
        self.sem_path_list = sem_path_list
        self.depth_path_list = depth_path_list
        self.train_sem_path_list = train_sem_path_list

        self.simulation_transform = transform[0] if transform is not None else None
        self.infer_transform = transform[1] if transform is not None else None
        self.train_transform = transform[2] if transform is not None else None
        self.aux = aux

    def embedding(self, case):

        assert case in [
            'Case_1', 
            'Case_2', 
            'Case_3', 
            'Case_4', 
            'Depth_110',
            'Depth_120',
            'Depth_130',
            'Depth_140',
        ]

        emb_dict = {
            'Case_1': 0,
            'Case_2': 1,
            'Case_3': 2,
            'Case_4': 3,
            'Depth_110': 0,
            'Depth_120': 1,
            'Depth_130': 2,
            'Depth_140': 3,
        }
        return emb_dict[case]
        

    def get_train_idx(self):
        train_data_len = len(self.train_sem_path_list)
        return random.randint(0, train_data_len-1)

    def __getitem__(self, idx):
        sem_path = self.sem_path_list[idx]
        sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE) / 255.
        
        train_sem_path = self.train_sem_path_list[self.get_train_idx()]
        train_sem_img = cv2.imread(train_sem_path, cv2.IMREAD_GRAYSCALE) / 255.

        # train
        if self.depth_path_list is not None:
            
            simulation_label = sem_path.split('/')[-3]
            simulation_label = self.embedding(simulation_label) if self.aux else -1
            train_label = train_sem_path.split('/')[-3]
            train_label = self.embedding(train_label) if self.aux else -1

            depth_path = self.depth_path_list[idx]
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE) / 255.

            if self.simulation_transform:
                transformed = self.simulation_transform(image=sem_img, mask=depth_img)
                sem_img = transformed['image']

                depth_img = transformed['mask'].unsqueeze(dim=0)

                train_sem_img = self.train_transform(image=train_sem_img)['image']

            else:
                transformed = ToTensorV2()
                sem_img = transformed(image=sem_img)['image']
                depth_img = transformed(image=depth_img)['image']
                train_sem_img = transformed(image=train_sem_img)['image']

            return (
                (sem_path, depth_path, train_sem_path), 
                sem_img.float(), 
                depth_img.float(), 
                simulation_label,
                train_sem_img.float(),
                train_label, 
                )

        # infer
        else:
            if self.infer_transform:
                transformed = self.infer_transform(image=sem_img)
                sem_img = transformed['image']
            else:
                transform = ToTensorV2()
                sem_img = transform(image=sem_img)['image']

            return sem_path, sem_img.float()


    def __len__(self):
        return len(self.simulation_sem_path_list)
