import os
import time
import random
import zipfile
import hydra
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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


def log_predictions(id, x, y, pred):

    import wandb

    x = unnormalize(x)
    y = y * 255.
    pred = pred * 255
    cat = np.concatenate([x, y, pred], axis=2)
    cat = np.transpose(cat , (1, 2, 0))

    images = wandb.Image(cat, caption=f"{id} (SEM / DEPTH / OUTPUT)")
    wandb.log({"examples": images})

def log_infer_results(id, x, pred):

    import wandb

    x = unnormalize(x)
    cat = np.concatenate([x, pred], axis=2)
    cat = np.transpose(cat , (1, 2, 0))

    images = wandb.Image(cat, caption=f"{id} (SEM / OUTPUT)")
    wandb.log({"examples": images})