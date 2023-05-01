import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import collections
import numpy as np


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        if args.dataset == 'cifar10':
            self.z_dim = 256
            self.hidden_dim = 1024
            self.output_dim = 3*32*32
            self.noise_dim = 64
        elif args.dataset == 'mnist' or args.dataset=='fashion' or args.dataset == 'femnist':
            self.z_dim = 32
            self.hidden_dim = 256
            self.output_dim = 1*28*28
            self.noise_dim = 32
        #
        self.crossentropy_loss=nn.NLLLoss(reduce=False).to(self.device)
        self.diversity_loss = DiversityLoss(metric='l1').to(self.device)
        self.dist_loss = nn.MSELoss().to(self.device)
        #
        input_dim = self.noise_dim + self.num_classes
        self.fc_layers = nn.ModuleList()
        fc = nn.Linear(input_dim, self.hidden_dim)
        bn = nn.BatchNorm1d(self.hidden_dim)
        act = nn.ReLU()
        representation_layer = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc_layers += [fc, bn, act, representation_layer]
 
    def forward(self, labels):
        labels = labels.clone().detach().to('cpu')
        result = {}
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim))
        result['eps'] = eps
        y_input = torch.FloatTensor(batch_size, self.num_classes)
        y_input.zero_()
        y_input.scatter_(1, labels.view(-1,1), 1)
        z = torch.cat((eps, y_input), dim=1)
        for layer in self.fc_layers:
            z = layer(z)
        result['output'] = z
        return result



class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))