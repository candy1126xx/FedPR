from Local.server import LocalServer
from FedAvg.user import AvgUser


class AvgServer(LocalServer):
    def __init__(self, args):
        super(AvgServer, self).__init__(args)


    def create_client(self, args, client_id, model):
        return AvgUser(args, self, client_id, model, self.train_dataset, self.test_dataset, self.train_groups[client_id], self.test_groups[client_id], self.label_counts[client_id])


    def proceed(self):
        self.evaluate(-1)
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n")
            self.selected_users, self.user_idxs = self.select_users(glob_iter, self.per_users)
            self.aggregate_parameters()
            self.send_parameters()
            self.local_train(glob_iter)
            self.evaluate(glob_iter)
        return self.mertrics
        

    def aggregate_parameters(self):
        for model_name, group in self.groups.items():
            total_train_samples = 0
            for user in group.users:
                total_train_samples += user.train_samples
            for server_param in group.global_model.parameters():
                server_param.data.zero_()
            for user in group.users:
                ratio = user.train_samples / total_train_samples
                for server_param, user_param in zip(group.global_model.parameters(), user.model.parameters()):
                    server_param.data += ratio * user_param.data


    def send_parameters(self):
        for model_name, group in self.groups.items():
            for user in group.users:
                for old_param, new_param in zip(user.model.parameters(), group.global_model.parameters()):
                    old_param.data = new_param.data.clone().detach()
