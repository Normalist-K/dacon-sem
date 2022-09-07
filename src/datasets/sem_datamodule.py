from copy import copy, deepcopy
import os
from typing import Optional
from glob import glob

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from src.datasets.sem_dataset import SEMDataset
from src.datasets.transform import get_transform, get_transform1, get_transform2


class SEMDataModule():
    def __init__(
        self,
        data_path: str = '/shared/Samsung',
        n_splits: int = 5,
        fold: int = 0,
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        verbose: bool = False,
        aug: int = 0,
        resize: list = [96, 64],
        aux: bool = False
    ):

        self.data_path = data_path
        self.n_splits = n_splits
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.verbose = verbose
        self.aug = int(aug)
        self.resize = resize
        self.aux = aux

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def set_cfg(self, cfg):
        self.cfg = cfg

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""

        if not os.path.exists(self.data_path):
            print(f"No Data in {self.data_path}.")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        data_path = os.path.abspath(self.data_path)
        simulation_sem_paths = os.path.join(data_path, 'simulation_data', 'SEM', '*', '*', '*.png')
        simulation_sem_paths = np.array(sorted(glob(simulation_sem_paths)))
        simulation_depth_paths = os.path.join(data_path, 'simulation_data', 'Depth', '*', '*', '*.png')
        simulation_depth_paths = np.array(sorted(glob(simulation_depth_paths) + glob(simulation_depth_paths)))
        data_len = len(simulation_sem_paths)

        skf = StratifiedKFold(n_splits=self.n_splits, random_state=self.cfg.seed, shuffle=True)
        splitlist = list(skf.split(range(data_len),[0]*data_len))

        train_index = splitlist[self.fold][0]
        valid_index = splitlist[self.fold][1]
        
        if self.verbose:
            print(f'Fold {self.fold} : train {len(train_index)}, valid {len(valid_index)}')

        train_sem_paths = simulation_sem_paths[train_index]
        train_depth_paths = simulation_depth_paths[train_index]
        
        valid_sem_paths = simulation_sem_paths[valid_index]
        valid_depth_paths = simulation_depth_paths[valid_index]

        test_sem_paths = os.path.join(data_path, 'test', 'SEM', '*.png')
        test_sem_paths = np.array(sorted(glob(test_sem_paths)))

        # transform = get_transform(self.resize) if self.aug else None
        if self.aug == 0:
            transform = get_transform(self.resize)
        elif self.aug == 1:
            transform = get_transform1(self.resize)
        elif self.aug == 2:
            transform = get_transform2(self.resize)

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val:

            if stage in (None, 'fit'):
                if self.cfg.small_dataset:
                    small_len = len(train_sem_paths) // 10
                    self.data_train = SEMDataset(deepcopy(train_sem_paths[:small_len]), deepcopy(train_depth_paths[:small_len]), transform, self.aux)
                else:
                    self.data_train = SEMDataset(train_sem_paths, train_depth_paths, transform, self.aux)
                self.data_val = SEMDataset(valid_sem_paths, valid_depth_paths, transform, self.aux)
                if self.verbose: print('train/val dataset loaded.')

        if not self.data_test:
            if stage in (None, 'predict'):
                self.data_test = SEMDataset(test_sem_paths, None, transform, False)
                if self.verbose: print('test dataset loaded.')


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
