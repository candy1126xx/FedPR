import torch, torch.nn as nn
from Local.user import LocalUser


class BABUUser(LocalUser):
    def __init__(self, args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts):
        super(BABUUser, self).__init__(args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts)
        self.optimizer = torch.optim.SGD(self.model.representor.parameters(), lr=args.learning_rate, momentum=0.5)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)
        self.model.freeze_pre()
