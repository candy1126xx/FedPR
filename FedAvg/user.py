
from Local.user import LocalUser
import torch, torch.nn as nn, numpy as np


class AvgUser(LocalUser):
    def __init__(self, args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts):
        super(AvgUser, self).__init__(args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts)
 