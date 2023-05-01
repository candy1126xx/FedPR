from FedAvg.server import AvgServer
from FedGen.user import GenUser
from FedGen.generator import Generator
import numpy as np, torch


class GenServer(AvgServer):
    def __init__(self, args):
        super(GenServer, self).__init__(args)
        self.gen_model = Generator(args)
        self.gen_optimizer = torch.optim.Adam(self.gen_model.parameters(), 
            lr=args.generator_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=1e-2, amsgrad=False)
        

    def create_client(self, args, client_id, model):
        return GenUser(args, self, client_id, model, self.train_dataset, self.test_dataset, self.train_groups[client_id], self.test_groups[client_id], self.label_counts[client_id])


    def proceed(self):
        self.evaluate(-1)
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n")
            self.selected_users, self.user_idxs = self.select_users(glob_iter, self.per_users)
            self.aggregate_parameters()
            self.send_parameters()
            self.local_train(glob_iter)
            self.evaluate(glob_iter)
            self.train(glob_iter)
        return self.mertrics

    
    # 训练生成模型
    def train(self, glob_iter):
        print("\n------- Server Training...... \n")
        self.gen_model.train()
        for i in range(5):
            y = np.random.choice([0,1,2,3,4,5,6,7,8,9], self.batch_size)
            y_input = torch.LongTensor(y).to(self.device)
            self.gen_optimizer.zero_grad()
            gen_result = self.gen_model(y_input)
            gen_output, eps = gen_result['output'].to(self.device), gen_result['eps'].to(self.device)
            diversity_loss = self.gen_model.diversity_loss(eps, gen_output)
            # teacher loss
            teacher_loss = 0
            for user in self.selected_users:
                weight = self.label_weights[user.id].clone().to(self.device)
                user.model.eval()
                logits, scores = user.model.forward_rep(gen_output)
                teacher_loss_ = torch.mean(self.gen_model.crossentropy_loss(scores, y_input) * weight[y])
                teacher_loss += teacher_loss_
            loss = teacher_loss + diversity_loss
            loss.backward()
            self.gen_optimizer.step()
