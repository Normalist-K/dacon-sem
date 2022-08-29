import torch
import torch.nn as nn


def rmse(pred, y, device):
    mse = nn.MSELoss().to(device)
    # pred = (pred*255.).type(torch.int8)
    # y = (y*255.).type(torch.int8)
    return torch.sqrt(mse(pred*255, y*255))