from FedAvg.server import AvgServer
from FedRep.user import RepUser


class RepServer(AvgServer):
    def __init__(self, args):
        super(RepServer, self).__init__(args)


    def create_client(self, args, client_id, model):
        return RepUser(args, self, client_id, model, self.train_dataset, self.test_dataset, self.train_groups[client_id], self.test_groups[client_id], self.label_counts[client_id])
        

    def send_parameters(self):
        for user in self.selected_users:
            for old_param, new_param in zip(user.model.representor.parameters(), self.global_model.representor.parameters()):
                old_param.data = new_param.data.clone().detach()
