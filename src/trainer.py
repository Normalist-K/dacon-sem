import os
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import albumentations as A
import wandb
from warmup_scheduler import GradualWarmupScheduler

from src.models.components.model_ema import ModelEMA
from src.utils.metric import rmse
from src.utils.utils import log_infer_results, log_predictions, unnormalize


class Trainer:
    def __init__(self, cfg, model, device, verbose=True):
        self.cfg = cfg
        self.model = model.to(device)
        if cfg.trainer.model_ema:
            self.model_ema = ModelEMA(self.model)
        self.device = device
        self.verbose = verbose

        # Loss function
        if cfg.trainer.criterion == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss(reduction='none')
        elif cfg.trainer.criterion == 'MSELoss':
            criterion = nn.MSELoss()
        elif cfg.trainer.criterion == 'L1Loss':
            criterion = nn.L1Loss()

        # Optimizer
        if cfg.trainer.optimizer.name.lower() == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(),
                                    lr=cfg.trainer.optimizer.lr,
                                    amsgrad=cfg.trainer.optimizer.amsgrad,
                                    weight_decay=cfg.trainer.optimizer.weight_decay)
        elif cfg.trainer.optimizer.name.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=cfg.trainer.optimizer.lr,
                                   weight_decay=cfg.trainer.optimizer.weight_decay)
        elif cfg.trainer.optimizer.name.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=cfg.trainer.optimizer.lr,
                                  weight_decay=cfg.trainer.optimizer.weight_decay,
                                  momentum=cfg.trainer.optimizer.momentum)

        # Scheduler
        if cfg.trainer.scheduler.name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             'min',
                                                             factor=cfg.trainer.scheduler.lr_factor,
                                                             patience=cfg.trainer.scheduler.patience,
                                                             threshold_mode= cfg.trainer.scheduler.threshold_mode,
                                                             cooldown= cfg.trainer.scheduler.cooldown,
                                                             min_lr= cfg.trainer.scheduler.min_lr,
                                                             eps= cfg.trainer.scheduler.eps,
                                                             verbose=verbose)
        elif cfg.trainer.scheduler.name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                             T_max=(cfg.len_train_loader * cfg.epoch),
                                                             eta_min=cfg.trainer.scheduler.min_lr)
        else:
            scheduler = None

        if scheduler is not None and cfg.trainer.scheduler.warmup:
            scheduler = GradualWarmupScheduler(optimizer, 
                                               multiplier=1, 
                                               total_epoch=3*cfg.len_train_loader, 
                                               after_scheduler=scheduler)

        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        if verbose:
            print('==Criterion==')
            print(self.criterion)
            print('==Optimizer==')
            print(self.optimizer)
            print('==Scheduler==')
            print(self.scheduler)

        self.start_epoch = 0
        if cfg.resume:
            checkpoint = torch.load(cfg.path.pretrained)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            if verbose:
                print(f"Model loaded: {cfg.path.pretrained}")
        
        self.best_model = deepcopy(self.model)
        self.best_valid_metric = np.Inf
        self.es_patience = 0
        self.scaler = GradScaler() if cfg.mixed_precision else None
        
    def fit(self, train_loader, valid_loader):
        
        # 1 epoch
        for epoch in range(self.start_epoch, self.start_epoch + self.cfg.epoch):
            
            # Train
            train_loss = self.train_epoch(train_loader)
            print(f'\nEpoch: {epoch}')
            print(f'Train Loss: {train_loss:.6f}')

            # Validation
            valid_loss, valid_metric = self.validation(valid_loader)
            print(f'Valid Loss: {valid_loss:.6f}, Valid metric: {valid_metric:.6f}')
            
            if self.cfg.trainer.scheduler.name == 'ReduceLROnPlateau':
                self.scheduler.step(valid_loss)
            
            # Log
            if not self.cfg.DEBUG:
                wandb.log({"train_loss": train_loss,
                           "valid_loss": valid_loss, 
                           "valid_metric": valid_metric,
                           "lr": self.optimizer.param_groups[0]['lr'],
                           })
        
            # Model EMA & Early stopping & Model save
            if valid_metric < self.best_valid_metric:
                if self.cfg.trainer.model_ema:
                    self.best_model = deepcopy(self.model_ema.ema)
                else:
                    self.best_model = deepcopy(self.model)
                self.best_valid_metric = valid_metric
                self.es_patience = 0
                if not self.cfg.DEBUG:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.best_model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'metric': valid_metric,
                        'loss': valid_loss,
                    }, os.path.join(self.cfg.path.weights, f'{self.cfg.name}-{self.cfg.dt_string}/{epoch}.pth')) 
                    print(f'Epoch {epoch} Model saved. ({self.cfg.name}-{self.cfg.dt_string}/{epoch}.pth)')
            elif epoch == (self.start_epoch + self.cfg.epoch - 1):
                if self.cfg.trainer.model_ema:
                    save_model = deepcopy(self.model_ema.ema)
                else:
                    save_model = deepcopy(self.model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': valid_loss,
                }, os.path.join(self.cfg.path.weights, f'{self.cfg.name}-{self.cfg.dt_string}/{epoch}_last.pth')) 
                print(f'Epoch {epoch} Model saved. ({self.cfg.name}-{self.cfg.dt_string}/{epoch}_last.pth)')
            else:
                self.es_patience += 1
                print(f"Valid metric. increased. Current early stop patience is {self.es_patience}")

            if (self.cfg.es_patience != 0) and (self.es_patience == self.cfg.es_patience):
                break

    def train_epoch(self, train_loader):
        self.model.train()

        losses = []
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, position=0, desc='Train')
        for batch_idx, (data_path, x, y, aux_y) in pbar:

            self.optimizer.zero_grad()
            
            x = x.to(self.device)
            y = y.to(self.device)
        
            if self.cfg.mixed_precision:
                with autocast():
                    pred = self.model(x)
                    loss = self.criterion(pred, y)

                self.scaler.scale(loss).backward()

                if self.cfg.trainer.optimizer.clipping:
                    # Gradient Norm Clipping
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()

                if self.cfg.trainer.optimizer.clipping:
                    # Gradient Norm Clipping
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
                
                self.optimizer.step()


            losses.append(loss.cpu().item())
            pbar.set_postfix(loss=loss.cpu().item())
            
            if self.cfg.trainer.model_ema:
                self.model_ema.update(self.model)

            if (self.scheduler is not None) and (self.cfg.trainer.scheduler.name != 'ReduceLROnPlateau'):
                self.scheduler.step()
            
            if self.cfg.DEBUG and batch_idx > 5:
                break
            
        return np.average(losses)

    def validation(self, valid_loader):
        if self.cfg.trainer.model_ema:
            model = self.model_ema.ema.eval()
        else:
            model = self.model.eval()

        losses, metrics = [], []
        p_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid', position=0, leave=True)
        for batch_idx, (data_path, x, y, aux_y) in p_bar:
            
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                pred = self.model(x)

                loss = self.criterion(pred, y)
                losses.append(loss.cpu().item())
                
                metric = rmse(pred, y)
                metrics.append(metric.cpu().item())
                
                # check results
                if not self.cfg.DEBUG and batch_idx == 0:
                    log_predictions(
                        os.path.basename(data_path[0][0]),
                        x[0].detach().cpu().numpy(),
                        y[0].detach().cpu().numpy(), 
                        pred[0].detach().cpu().numpy()
                    )

        return np.average(losses), np.average(metrics)

    def inference(self, test_loader):
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        
        result_names = []
        result_preds = []

        p_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Infer', position=0, leave=True)
        for batch_idx, (batch_path, batch_x) in p_bar:
            batch_x = batch_x.to(self.device)

            with torch.no_grad():
                batch_pred = self.best_model(batch_x)

                batch_pred = batch_pred * 255.

                # check results
                if not self.cfg.DEBUG and batch_idx == 0:
                    log_infer_results(
                        os.path.basename(batch_path[0]),
                        batch_x[0].detach().cpu().numpy().copy(),
                        batch_pred[0].detach().cpu().numpy().copy()
                    )

                for path, x, pred in zip(batch_path, batch_x, batch_pred):
                    name = os.path.basename(path)

                    pred = pred.squeeze().cpu().numpy()
                    pred = A.Resize(*self.cfg.original_img_size, always_apply=True)(image=pred)['image']

                    result_names.append(name)
                    result_preds.append(pred)

        return result_names, result_preds