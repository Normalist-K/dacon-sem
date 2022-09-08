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
from src.utils.utils import log_infer_results, log_predictions, unnormalize, log_heatmap


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
                                    weight_decay=cfg.trainer.optimizer.weight_decay)
        elif cfg.trainer.optimizer.name.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=cfg.trainer.optimizer.lr,
                                    amsgrad=cfg.trainer.optimizer.amsgrad,
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

        if self.cfg.model.aux:
            self.aux_criterion = nn.CrossEntropyLoss().to(self.device)
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
            train_loss, train_depth_loss, train_aux_loss = self.train_epoch(train_loader)
            print(f'\nEpoch: {epoch}')
            print(f'Train Loss: {train_loss:.6f}')

            # Validation
            valid_loss, valid_metric, valid_depth_loss, valid_aux_loss, valid_acc = self.validation(valid_loader)
            print(f'Valid Loss: {valid_loss:.6f}, Valid metric: {valid_metric:.6f}, Valid acc: {valid_acc:.2f}')
            
            if self.cfg.trainer.scheduler.name == 'ReduceLROnPlateau':
                self.scheduler.step(valid_loss)
            
            # Log
            if not self.cfg.DEBUG:
                wandb.log({
                    "train": {
                        "loss": train_loss,
                        "depth_loss": train_depth_loss,
                        "aux_loss": train_aux_loss,
                        "lr": self.optimizer.param_groups[0]['lr'],
                    },
                    "valid": {
                        "loss": valid_loss, 
                        "metric": valid_metric,
                        "depth_loss": valid_depth_loss,
                        "aux_loss": valid_aux_loss,
                        "acc": valid_acc,
                    },
                })           
        
            # Model EMA & Early stopping & Model save
            if valid_metric < self.best_valid_metric:
                if self.cfg.trainer.model_ema:
                    self.best_model = deepcopy(self.model_ema.ema)
                else:
                    self.best_model = deepcopy(self.model)
                self.best_valid_metric = valid_metric
                self.best_valid_preds = self.valid_preds
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

    def step_(self, x, y, aux_label):

        self.model.check_input_shape(x)

        features = self.model.encoder(x)
        decoder_output = self.model.decoder(*features)

        pred = self.model.segmentation_head(decoder_output)

        if self.model.classification_head is not None:
            aux_pred = self.model.classification_head(features[-1])

        pred, aux_pred = self.model(x)
        depth_loss = self.criterion(pred, y)
        aux_loss = self.aux_criterion(aux_pred, aux_label)
        loss = self.cfg.trainer.loss_alpha * depth_loss + aux_loss

        return {
            "pred": pred,
            "aux_pred" : aux_pred, 
            "loss" : loss, 
            "depth_loss" : depth_loss, 
            "aux_loss" : aux_loss
            }
            
    def step(self, x, y, aux_label):
        pred, aux_pred = self.model(x)
        depth_loss = self.criterion(pred, y)
        aux_loss = self.aux_criterion(aux_pred, aux_label)
        loss = self.cfg.trainer.loss_alpha * depth_loss + aux_loss

        return {
            "pred": pred,
            "aux_pred" : aux_pred, 
            "loss" : loss, 
            "depth_loss" : depth_loss, 
            "aux_loss" : aux_loss
            }

    def train_epoch(self, train_loader):
        self.model.train()

        losses = []
        depth_losses = []
        aux_losses = []
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, position=0, desc='Train')
        for batch_idx, (data_path, sem, depth, aux_label, train_sem, train_aux_label) in pbar:

            self.optimizer.zero_grad()
            
            sem = sem.to(self.device)
            depth = depth.to(self.device)
            aux_label = aux_label.to(self.device)
        
            if self.cfg.mixed_precision:
                with autocast():
                    step_out = self.step(sem, depth, aux_label)
                    aux_loss = step_out['aux_loss']
                    depth_loss = step_out['depth_loss']
                    loss = step_out['loss']

                self.scaler.scale(loss).backward()

                if self.cfg.trainer.optimizer.clipping:
                    # Gradient Norm Clipping
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                step_out = self.step(sem, depth, aux_label)
                aux_loss = step_out['aux_loss']
                depth_loss = step_out['depth_loss']
                loss = step_out['loss']

                loss.backward()

                if self.cfg.trainer.optimizer.clipping:
                    # Gradient Norm Clipping
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
                
                self.optimizer.step()

            depth_losses.append(depth_loss.cpu().item())
            aux_losses.append(aux_loss.cpu().item())
            losses.append(loss.cpu().item())
            pbar.set_postfix(loss=loss.cpu().item())
            
            if self.cfg.trainer.model_ema:
                self.model_ema.update(self.model)

            if (self.scheduler is not None) and (self.cfg.trainer.scheduler.name != 'ReduceLROnPlateau'):
                self.scheduler.step()
            
        return np.average(losses), np.average(depth_losses), np.average(aux_losses)

    def validation(self, valid_loader):
        if self.cfg.trainer.model_ema:
            model = self.model_ema.ema.eval()
        else:
            model = self.model.eval()

        losses, metrics = [], []
        depth_losses = []
        aux_losses = []
        correct, total = 0, 0
        p_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid', position=0, leave=True)
        
        preds = []

        for batch_idx, (data_path, sem, depth, aux_label, train_sem, train_aux_label) in p_bar:
            
            sem = sem.to(self.device)
            depth = depth.to(self.device)
            aux_label = aux_label.to(self.device)

            with torch.no_grad():
                step_out = self.step(sem, depth, aux_label)
                pred = step_out['pred']
                aux_pred = step_out['aux_pred']
                aux_loss = step_out['aux_loss']
                depth_loss = step_out['depth_loss']
                loss = step_out['loss']

                preds.append(pred*255.)
                depth_losses.append(depth_loss.cpu().item())
                aux_losses.append(aux_loss.cpu().item())
                losses.append(loss.cpu().item())
                
                metric = rmse(pred, depth, self.device)
                metrics.append(metric.cpu().item())

                _, predicted = torch.max(aux_pred.data, 1)
                total += aux_label.size(0)
                correct += (predicted == aux_label).sum().item()
                
                # check results
                if not self.cfg.DEBUG and batch_idx == 0:
                    log_predictions(
                        os.path.basename(data_path[0][0]),
                        sem[0].detach().cpu().numpy(),
                        depth[0].detach().cpu().numpy(), 
                        pred[0].detach().cpu().numpy()
                    )
        accuracy = correct / total * 100

        self.valid_preds = torch.cat(preds, dim=0).squeeze().detach().cpu().numpy()

        return np.average(losses), np.average(metrics), np.average(depth_losses), np.average(aux_losses), accuracy

    def inference(self, test_loader):
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        
        result_names = []
        result_preds = []

        p_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Infer', position=0, leave=True)
        for batch_idx, (batch_path, batch_sem) in p_bar:
            batch_sem = batch_sem.to(self.device)

            with torch.no_grad():
                batch_pred, _ = self.best_model(batch_sem)

                batch_pred = batch_pred * 255.

                # check results
                if not self.cfg.DEBUG and batch_idx == 0:
                    log_infer_results(
                        os.path.basename(batch_path[0]),
                        batch_sem[0].detach().cpu().numpy().copy(),
                        batch_pred[0].detach().cpu().numpy().copy()
                    )

                for path, x, pred in zip(batch_path, batch_sem, batch_pred):
                    name = os.path.basename(path)

                    pred = pred.squeeze().cpu().numpy()
                    # if self.cfg.datamodule.aug:
                    pred = A.Resize(*self.cfg.original_img_size, always_apply=True)(image=pred)['image']
                        

                    result_names.append(name)
                    result_preds.append(pred)

        preds = np.concatenate(result_preds, axis=0)
        preds = preds.reshape(len(result_preds), 72, 48)
        if not self.cfg.DEBUG:
            # if hasattr(self, 'best_valid_preds'):
            #     log_heatmap(self.best_valid_preds, self.cfg.name, 'valid')
            log_heatmap(preds, self.cfg.name, 'test')

        return result_names, result_preds