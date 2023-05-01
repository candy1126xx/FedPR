from FedAvg.server import AvgServer
from FedProx.user import ProxUser


class ProxServer(AvgServer):
    def __init__(self, args):
        super(ProxServer, self).__init__(args)

    def create_client(self, args, client_id, model):
        return ProxUser(args, self, client_id, model, self.train_dataset, self.test_dataset, self.train_groups[client_id], self.test_groups[client_id], self.label_counts[client_id])

    def send_parameters(self):
        for model_name, group in self.groups.items():
            for user in group.users:
                for old_param, new_param in zip(user.model.parameters(), group.global_model.parameters()):
                    old_param.data = new_param.data.clone().detach()
                for old_param, new_param in zip(user.global_model.parameters(), group.global_model.parameters()):
                    old_param.data = new_param.data.clone().detach()