import torch, torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_utils import DatasetSplit
from utils.models import create_model


class LocalUser:
    def __init__(self, args, server, client_id, local_model_name, train_dataset, test_dataset, train_idx, test_idx, label_counts):
        # 参数
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.dataset = args.dataset
        self.device = args.device
        self.num_classes = args.num_classes
        self.gamma = args.gamma
        self.algorithm = args.algorithm
         # 客户端ID
        self.id = client_id
        self.server = server
        # 创建模型
        self.model_name, self.model = create_model(local_model_name)
        self.model.to(self.device)
        # 训练集样本数
        self.train_samples = len(train_idx)
        # 测试集样本数
        self.test_samples = len(test_idx)
        self.label_counts = label_counts
        self.total_label_counts = torch.sum(self.label_counts, dim=0)
        self.available_labels = [i for i in range(len(label_counts)) if label_counts[i] > 0]
        # DataLoader
        self.trainloader = DataLoader(DatasetSplit(train_dataset, train_idx), batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.testloader = DataLoader(DatasetSplit(test_dataset, test_idx), batch_size=10, shuffle=False, drop_last=True)
        # 初始化损失函数
        self.predict_loss = nn.CrossEntropyLoss().to(args.device)
        # 本地模型的优化器
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.5)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)


    def train(self, glob_iter):
        self.model.train()
        loss = 0.0
        for batch_idx, (x, y) in enumerate(self.trainloader, 0):
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            loss += self._train(glob_iter, x, y, self.model)
            self.optimizer.step()
        self.scheduler.step()
        return loss

    
    def _train(self, glob_iter, x, y, model):
        representation, logits, scores = model(x)
        loss = self.predict_loss(scores, y)
        loss.backward()
        return loss.item()


    def test(self):
        if self.algorithm=='FedPR' or self.algorithm=='pFedPR+':
            acc = self.get_acc_by_local_prototype(self.model)
        else:
            acc = self.get_acc_with_head(self.model)
        return acc


    def get_acc_with_head(self, test_model):
        total, correct = 0, 0
        test_model.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.testloader, 0):
                x, y = x.to(self.device), y.to(self.device)
                representation, logits, scores = test_model(x)
                total += y.shape[0]
                _, predicted = scores.max(1)
                correct += predicted.eq(y).sum().item()
        return 100.*correct/total


    def get_acc_by_local_prototype(self, test_model):
        total, correct = 0, 0
        test_model.eval()
        representation_dict = {}
        cos_sim = nn.CosineSimilarity(dim=0).to(self.device)
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.trainloader, 0):
                x, y = x.to(self.device), y.to(self.device)
                representations, logits, scores = test_model(x)
                for i in range(y.shape[0]):
                    if y[i].item() in representation_dict.keys():
                        representation_dict[y[i].item()].append(representations[i].clone().detach())
                    else:
                        representation_dict[y[i].item()] = [representations[i].clone().detach()]
            local_prototype = {}
            for label, representations in representation_dict.items():
                if len(representations) > 1:
                    representation_sum = torch.zeros_like(representations[0]).to(self.device)
                    for i in representations:
                        representation_sum += i.data
                    local_prototype[label] = representation_sum / len(representations)
                else:
                    local_prototype[label] = representations[0].data
            #
            for batch_idx, (x, y) in enumerate(self.testloader, 0):
                x, y = x.to(self.device), y.to(self.device)
                representation, logits, scores = test_model(x)
                dist = torch.zeros(size=(representation.shape[0], self.num_classes)).to(self.device)
                for i in range(representation.shape[0]):
                    for j in self.available_labels:
                        dist[i, j] = cos_sim(representation[i], local_prototype[j])
                total += y.shape[0]
                _, predicted = dist.max(1)
                correct += predicted.eq(y).sum().item()
        return 100.*correct/total
    