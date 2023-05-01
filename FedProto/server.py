
from Local.server import LocalServer
from FedProto.user import ProtoUser
import numpy as np, torch, torch.nn as nn, copy


class ProtoServer(LocalServer):
    def __init__(self, args):
        super(ProtoServer, self).__init__(args)
        self.glob_prototype = {}


    def create_client(self, args, client_id, model):
        return ProtoUser(args, self, client_id, model, self.train_dataset, self.test_dataset, self.train_groups[client_id], self.test_groups[client_id], self.label_counts[client_id])


    def proceed(self):
        self.evaluate(-1)
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n")
            self.selected_users, self.user_idxs = self.select_users(glob_iter, self.per_users)
            self.aggregate_prototype()
            self.local_train(glob_iter)
            self.evaluate(glob_iter)
        return self.mertrics


    def aggregate_prototype(self):
        proto_dict = {}
        self.glob_prototype = {}
        for user in self.selected_users:
            for label in user.local_prototype.keys():
                if label in proto_dict.keys():
                    proto_dict[label].append(user.local_prototype[label])
                else:
                    proto_dict[label] = [user.local_prototype[label]]
        for label, proto_list in proto_dict.items():
            if len(proto_list) > 1:
                proto = torch.zeros_like(proto_list[0]).to(self.device)
                for i in proto_list:
                    proto += i.data
                self.glob_prototype[label] = proto / len(proto_list)
            else:
                self.glob_prototype[label] = proto_list[0].clone().detach()
