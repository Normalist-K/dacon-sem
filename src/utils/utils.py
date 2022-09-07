import os
from glob import glob
import time
import random
import zipfile
import hydra
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb



def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def load_dataloader(cfg):
    print(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.set_cfg(cfg)
    dm.prepare_data()
    dm.setup(stage=None)
    train_loader = dm.train_dataloader()
    valid_loader = dm.val_dataloader()
    test_loader = dm.predict_dataloader()

    if cfg.trainer.scheduler.name == 'CosineAnnealingLR':
        cfg.len_train_loader = len(train_loader)

    return train_loader, valid_loader, test_loader


def save_submission(cfg, results):
    result_names, result_preds = results

    save_path = cfg.path.submissions
    timestr = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    os.chdir(save_path)
    sub_imgs = []
    for name, pred_img in zip(result_names, result_preds):
        cv2.imwrite(name, pred_img)
        sub_imgs.append(name)
    with zipfile.ZipFile(f"../submission-{timestr}.zip", 'w') as sub:
        for name in sub_imgs:
            sub.write(name)
    print(f'submission-{timestr}.zip saved.')


def get_png_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, axis=-1)
    return img


def unnormalize(images):
        return (0.5*np.array(images)+0.5)*255

def draw_line_by_row(
        preds,
        save_dir_name='/home/youngkim21/dacon/dacon-sem/output/heatmap', 
        file_name='heatmap.png',
        stage='valid'
    ):

    from matplotlib import cm
    from matplotlib.colors import ListedColormap, BoundaryNorm


    if stage=='valid':
        height = 96
        width = 64
    elif stage=='test':
        height = 72
        width = 48
    
    pixel_tables=np.zeros((height, width, len(preds)),dtype=np.uint8)
    
    for i in range(0, len(preds)):
        img = preds[i]
        pixel_tables[:,:,i] = img.copy()
    
    
    heat_map = np.zeros((256,width,height))
    for i in range(0,pixel_tables.shape[0]):
        for j in range(0,pixel_tables.shape[1]):
            t_unique, t_counts = np.unique(pixel_tables[i,j,:],return_counts=True)
            for t_height,t_value in zip(t_unique,t_counts):
                heat_map[t_height,j,i]=t_value
    
    bounds = np.append(1,np.arange(1000,30001,2000))
    jet = cm.get_cmap('jet', 256)
    newcolors = jet(np.linspace(0, 1, 256))
    black = np.array([0/256, 0/256, 0/256, 1])
    #newcolors[:1, :] = black #colorbar customize
    newcmp = ListedColormap(newcolors)
    newcmp.set_over(newcolors[-1])
    newcmp.set_under(black)
    norm = BoundaryNorm(bounds, newcmp.N)
    
    fig=plt.figure(figsize=(34,20),constrained_layout=True,)
    
    subplot = fig.subplots(nrows=3, ncols=24)
    
    #plt.subplots_adjust(left=0.05,right=1,top=1,bottom=0.2,wspace=0.3,hspace=0.1)
    
    i=0
    for ax in subplot.flat:
        im_heatmap = ax.pcolor(heat_map[:,:,i],cmap=newcmp,#edgecolors='black',
                               vmin=0,vmax=heat_map.max(),norm=norm)
        i += 1
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    cb_ax=fig.add_axes([0.25,0.025,0.5,0.05])  ##<= (left,bottom,width,height)
    plt.colorbar(im_heatmap,cax=cb_ax,orientation='horizontal',pad=0.02,ticks=bounds,extend='both',norm=norm)
    
    os.makedirs(save_dir_name, exist_ok=True)
    heat_img_file_path = os.path.join(save_dir_name,file_name)
    plt.savefig(heat_img_file_path,dpi=fig.dpi)
    
    plt.cla()
    plt.clf()
    plt.close()


def log_predictions(id, x, y, pred):

    x = unnormalize(x)
    y = y * 255.
    pred = pred * 255
    cat = np.concatenate([x, y, pred], axis=2)
    cat = np.transpose(cat , (1, 2, 0))

    images = wandb.Image(cat, caption=f"{id} (SEM / DEPTH / OUTPUT)")
    wandb.log({"examples": images}, commit=False)

def log_heatmap(preds, name, stage):

    save_dir_name='/home/youngkim21/dacon/dacon-sem/output/heatmap'
    file_name=f'{name}.png'

    draw_line_by_row(preds, save_dir_name, file_name, stage)
    img = cv2.imread(os.path.join(save_dir_name, file_name))

    images = wandb.Image(img, caption=f"{stage} heatmap")
    wandb.log({"examples": images})

def log_infer_results(id, x, pred):

    x = unnormalize(x)
    cat = np.concatenate([x, pred], axis=2)
    cat = np.transpose(cat , (1, 2, 0))

    images = wandb.Image(cat, caption=f"{id} (SEM / OUTPUT)")
    wandb.log({"examples": images})