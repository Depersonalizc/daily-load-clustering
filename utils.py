import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy as sp
import numpy as np
import torch
from torch import nn, functional as F
from torch.utils.data import Dataset, DataLoader
# Dataset
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

class AE_CNN(nn.Module):
    def __init__(self, cfg, load_dict=None):
        super().__init__()
        act = cfg['activation']
        d = cfg['latent_dim']
        
        # encoder
        enc = [
            # 1440 -> 288
            nn.Conv1d(1, 16, 5, padding=2),
            nn.MaxPool1d(kernel_size=5),
            act,
            
            # 288 -> 72
            nn.Conv1d(16, 32, 4, padding=2),
            nn.MaxPool1d(kernel_size=4),
            act,
            
            # 32*72
            nn.Flatten(),
            # 32*72 -> d (fully connected)
            nn.Linear(32*72, d),
#             nn.ReLU(),
        ]

        # decoder
        dec = [
            # d -> 32*72 (fully connected)
            nn.Linear(d, 32*72),
#             nn.ReLU(),
            # 72
            View((32, 72)),

            # 32 -> 96
            nn.Conv1d(32, 16, 4, padding=2),
            nn.Upsample(288, mode='linear'),
            act,
            
            # 96 -> 288
            nn.Conv1d(16, 1, 5, padding=2),
            nn.Upsample(1440, mode='linear'),
#             nn.Tanh()
            nn.Sigmoid(),
        ]
        
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)
        
        if load_dict is not None:
            self.load_state_dict(torch.load(load_dict))
        
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return latent, recon
        return latent


class AE_MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        encoder = []
        decoder = []
        act = cfg['activation']
        
        # encoder
        for i in range(len(cfg['encoder']) - 1):
            cin, cout = cfg['encoder'][i], cfg['encoder'][i+1]
            encoder.append(nn.Linear(cin, cout))
            encoder.append(act)

        # decoder
        for i in range(len(cfg['decoder']) - 1):
            cin, cout = cfg['decoder'][i], cfg['decoder'][i+1]
            decoder.append(nn.Linear(cin, cout))
            decoder.append(act)

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return latent, recon

class DS(Dataset):
    def __init__(self, data, sep, train=True):
        super().__init__()
        if train:
            self.data = torch.Tensor(data[:sep]).cuda()
        else:
            self.data = torch.Tensor(data[sep:]).cuda()
        self.data.unsqueeze_(1)
    
    def __getitem__(self, i):
        return self.data[i]
    
    def __len__(self):
        return self.data.shape[0]



def normalized(load):
    peak = load.max(axis=1)[:, None]
    trough = load.min(axis=1)[:, None]
    diff = peak - trough
    diff[diff == 0.] = 1.
    normalized = (load - trough) / diff
    return normalized

def decode(latent_vec: np.array, model) -> np.array:
    latent_tensor = torch.Tensor(latent_vec[None,:,None]).cuda()
    with torch.no_grad():
        decoded = model.decoder(latent_tensor.permute(0, 2, 1))
    return decoded.cpu().numpy()[0, 0, :]
