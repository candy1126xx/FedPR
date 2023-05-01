import math, torch
from Local.server import LocalServer
from FedAMP.user import AMPUser


class AMPServer(LocalServer):
    def __init__(self, args):
        super(AMPServer, self).__init__(args)
        self.alphaK = 10000 # 10000
        self.sigma = 10 # 1，10，100，1000，10000


    def create_client(self, args, client_id, model):
        return AMPUser(args, self, client_id, model, self.train_dataset, self.test_dataset, self.train_groups[client_id], self.test_groups[client_id], self.label_counts[client_id])


    def proceed(self):
        self.evaluate(-1)
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n")
            self.selected_users, self.user_idxs = self.select_users(glob_iter, self.per_users)
            self.local_train(glob_iter)
            self.evaluate(glob_iter)
            self.send_parameters()
        return self.mertrics
        

    def send_parameters(self):
        for model_name, group in self.groups.items():
            for user_i in group.users:
                coef = {}
                for user_j in group.users:
                    if user_i.id != user_j.id:
                        weights_i = user_i.model.weight_flatten()
                        weights_j = user_j.model.weight_flatten()
                        sub = (weights_i - weights_j).view(-1)
                        sub = torch.dot(sub, sub)
                        coef[user_j.id] = self.alphaK * self.e(sub)
                    else:
                        coef[user_j.id] = 0
                coef_self = 1.0
                for user_id in coef.keys():
                    coef_self -= coef[user_id]

                for param in user_i.client_u.parameters():
                    param.data = coef_self * param.data
                for user_j in group.users:
                    for param, param_j in zip(user_i.client_u.parameters(), user_j.model.parameters()):
                        param.data += coef[user_j.id] * param_j


    def e(self, x):
        return math.exp(-x/self.sigma)/self.sigma
