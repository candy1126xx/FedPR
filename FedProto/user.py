import torch, torch.nn as nn
from Local.user import LocalUser


class ProtoUser(LocalUser):
    def __init__(self, args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts):
        super(ProtoUser, self).__init__(args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts)
        self.dist_loss = nn.MSELoss().to(self.device)
        self.gamma = 1.0
        self.local_prototype = {}


    def train(self, glob_iter):
        representation_dict = {}
        self.model.to(self.device)
        self.model.train()
        loss_item = 0.0
        for batch_idx, (x, y) in enumerate(self.trainloader, 0):
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            representations, logits, scores = self.model(x)
            predict_loss = self.predict_loss(scores, y)
            if glob_iter==0:
                loss = predict_loss
            else:
                representation_new = torch.zeros_like(representations).to(self.device)
                for i in range(y.shape[0]):
                    representation_new[i] = self.server.glob_prototype[y[i].item()].data
                dist_loss = self.dist_loss(representations, representation_new)
                loss = predict_loss + self.gamma * dist_loss
            loss.backward()
            loss_item += loss.item()
            self.optimizer.step()
            # progress_bar(self.id, batch_idx, len(self.trainloader))
            # 保存每个 batch 的 Z，y
            for i in range(y.shape[0]):
                if y[i].item() in representation_dict.keys():
                    representation_dict[y[i].item()].append(representations[i].clone().detach())
                else:
                    representation_dict[y[i].item()] = [representations[i].clone().detach()]
        self.scheduler.step()
        # 用整个数据集的 Z，y 计算本地原型集
        self.local_prototype = {}
        for label, representations in representation_dict.items():
            if len(representations) > 1:
                representation_sum = torch.zeros_like(representations[0]).to(self.device)
                for i in representations:
                    representation_sum += i.data
                self.local_prototype[label] = representation_sum / len(representations)
            else:
                self.local_prototype[label] = representations[0].data
        return loss_item
