from FedAvg.user import AvgUser


class RepUser(AvgUser):
    def __init__(self, args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts):
        super(RepUser, self).__init__(args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts)
