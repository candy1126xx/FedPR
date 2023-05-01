
import torch, torch.nn as nn, copy
from Local.user import LocalUser
from FedPR.generator import VAE


class PRUser(LocalUser):
    def __init__(self, args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts):
        super(PRUser, self).__init__(args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts)
        self.optimizer = torch.optim.SGD(self.model.representor.parameters(), lr=args.learning_rate, momentum=0.5)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)
        self.model.freeze_pre()
        self.dist_loss = nn.MSELoss().to(self.device)
        self.iter_trainloader = iter(self.trainloader) # Magic
        self.samples_per_round=5 # 每轮采样10次
        self.iasg_avg_steps=2    # 每1个batch合并取均值


    def _train(self, glob_iter, x, y, model):
        representations, logits, scores = model(x)
        predict_loss = self.predict_loss(scores, y)
        if self.gamma > 0 and glob_iter > 0:
            self.server.gen_model.eval()
            _, X_global, _ = self.server.gen_model(representations)
            X_global = X_global.view(-1, x.shape[1], x.shape[2], x.shape[3])
            dist_loss = self.dist_loss(X_global, x)
            loss = predict_loss + self.gamma * dist_loss
        else:
            loss = predict_loss
        loss.backward()
        return loss.item()


    def sample_yZ(self):
        state = self.model.state_dict()
        def step_sgd(step):
            self.model.train()
            proto_dict = {}
            for s in range(step):
                self.optimizer.zero_grad()
                x, y = self.get_next_train_batch()
                x, y = x.to(self.device), y.to(self.device)
                representations, logits, scores = self.model(x)
                loss = self.predict_loss(scores, y)
                loss.backward()
                self.optimizer.step()
                for i in range(y.shape[0]):
                    if y[i].item() in proto_dict.keys():
                        proto_dict[y[i].item()].append(representations[i,:])
                    else:
                        proto_dict[y[i].item()] = [representations[i,:]]
            label_sample = {}
            for label, proto_list in proto_dict.items():
                if len(proto_list) > 1:
                    proto = 0 * proto_list[0].data
                    for i in proto_list:
                        proto += i.data
                    label_sample[label] = proto / len(proto_list)
                else:
                    label_sample[label] = proto_list[0].data
            return label_sample
        yZ = []
        for i in range(self.samples_per_round):
            label_sample = step_sgd(self.iasg_avg_steps)
            for label, sample in label_sample.items():
                yZ.append((sample.data, label))
        self.model.load_state_dict(state)
        return yZ


    def get_next_train_batch(self):
        try:
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return X, y
