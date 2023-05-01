from Local.server import LocalServer
from FedPR.generator import VAE
from FedPR.user import PRUser
from torch.utils.data import DataLoader
from utils.models import create_model
import torch, torch.nn as nn


class PRServer(LocalServer):
    def __init__(self, args):
        super(PRServer, self).__init__(args)
        self.generator_lr = args.generator_lr
        self.gen_loss = torch.nn.MSELoss().to(self.device)
        self.gen_model = VAE(args).to(self.device)
        self.gen_optimizer = torch.optim.Adam(self.gen_model.parameters(), 
            lr=args.generator_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=1e-2, amsgrad=False)


    def create_client(self, args, client_id, model):
        return PRUser(args, self, client_id, model, self.train_dataset, self.test_dataset, self.train_groups[client_id], self.test_groups[client_id], self.label_counts[client_id])


    def proceed(self):
        self.send_head()
        self.evaluate(-1)
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n")
            self.selected_users, self.user_idxs = self.select_users(glob_iter, self.per_users)
            self.aggregate_parameters()
            self.train(glob_iter)
            self.local_train(glob_iter)
            self.evaluate(glob_iter)
        return self.mertrics


    def send_head(self):
        _, global_model = create_model(self.global_model_name)
        global_model.to(self.device)
        if self.param_init=='kaiming_uniform':
            nn.init.kaiming_uniform_(global_model.predictor.weight)
        elif self.param_init=='orthogonal':
            nn.init.orthogonal_(global_model.predictor.weight)
        for user in self.users:
            for user_param, server_param in zip(user.model.predictor.parameters(), global_model.predictor.parameters()):
                user_param.data = server_param.data.clone().detach()


    def aggregate_parameters(self):
        yZ = []
        for user in self.selected_users:
            yZ += user.sample_yZ()
        self.yZ_loader = DataLoader(yZ, batch_size=self.batch_size, shuffle=True, drop_last=True)


    # 训练生成模型
    def train(self, glob_iter):
        print("\n------- Server Training...... \n")
        self.gen_model.train()
        for batch_idx, (Z, y) in enumerate(self.yZ_loader, 0):
            self.gen_optimizer.zero_grad()
            Z, y = Z.to(self.device), y.to(self.device)
            hat_X, mu, log_var = self.gen_model(Z)
            auz = torch.zeros_like(Z).to(self.device)
            for user in self.selected_users:
                user_label_weights = self.label_weights[user.id].clone().to(self.device)
                weights = user_label_weights[y]
                user.model.eval()
                uz, logits, scores = user.model(hat_X)
                auz.data += torch.mul(uz, weights.reshape(uz.shape[0], 1)).data
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = self.gen_loss(auz, Z) + KLD
            loss.backward()
            self.gen_optimizer.step()
