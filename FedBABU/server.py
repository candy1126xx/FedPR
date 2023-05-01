from FedAvg.server import AvgServer
from FedBABU.user import BABUUser


class BABUServer(AvgServer):
    def __init__(self, args):
        super(BABUServer, self).__init__(args)


    def create_client(self, args, client_id, model):
        return BABUUser(args, self, client_id, model, self.train_dataset, self.test_dataset, self.train_groups[client_id], self.test_groups[client_id], self.label_counts[client_id])
