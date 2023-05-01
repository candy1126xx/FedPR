import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import collections
import numpy as np


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.dataset = args.dataset
        if args.dataset.startswith('synthetic'):
            self.z_dim = 16
            self.hidden_dim = 32
            self.output_dim = 64
        elif args.dataset == 'cifar10':
            self.z_dim = 256
            self.hidden_dim = 1024
            self.output_dim = 3*32*32
        elif args.dataset == 'mnist' or args.dataset=='fashion' or args.dataset == 'femnist':
            self.z_dim = 32
            self.hidden_dim = 256
            self.output_dim = 1*28*28
        self.fc1 = nn.Linear(self.z_dim, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.output_dim)
 
    # Z -> N
    def encoder(self, z):
        h = F.relu(self.fc1(z))
        return self.fc21(h), self.fc22(h)

    # N -> X
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
 
    def forward(self, z):
        mu, log_var = self.encoder(z)
        x = self.sampling(mu, log_var)
        if self.dataset == 'mnist' or self.dataset=='fashion' or self.dataset == 'femnist':
            x = x.view(-1, 1, 28, 28)
        elif self.dataset == 'cifar10':
            x = x.view(-1, 3, 32, 32)
        return x, mu, log_var
