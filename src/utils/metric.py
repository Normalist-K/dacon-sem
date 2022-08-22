import torch
import torch.nn as nn


def rmse(pred, y):
    return torch.sqrt(nn.MSELoss()(pred*255., y*255.))