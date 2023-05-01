from Local.user import LocalUser
from utils.data_utils import read_data
import torch, numpy as np, copy


class Group:
    def __init__(self, model):
        self.global_model = copy.deepcopy(model)
        self.users = []


class LocalServer:
    def __init__(self, args):
        # Set up the main attributes
        self.algorithm = args.algorithm
        self.global_model_name = args.model_name
        self.param_init = args.param_init
        self.num_glob_iters = args.num_glob_iters
        self.local_ep = args.local_ep
        self.num_users = args.num_users
        self.per_users = args.per_users
        self.num_classes = args.num_classes
        self.device = args.device
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.num_partial_work = args.num_partial_work
        self.output_global_model = self.algorithm=='FedAvg' or self.algorithm=='FedProx' or self.algorithm=='Scaffold' or self.algorithm=='FedPR+' or self.algorithm=='FedBABU'
        # 所有客户端
        self.users = []
        # 1 轮通信中被选择的客户端
        self.selected_users = []
        # clients: [user_id]
        # train_dataset, test_dataset: Dataset
        # train_groups, test_groups: {user_id: [sample_ids]}
        # label_counts: [user_id][label] = 样本数量
        self.client_ids, self.train_dataset, self.test_dataset, self.train_groups, self.test_groups, self.label_counts = read_data(args)
        # 有多少个客户端
        self.total_users = len(self.client_ids)
        # 某类训练样本的总量
        self.total_label_counts = torch.sum(self.label_counts, dim=0)
        # 训练样本总量
        self.total_train_samples = torch.sum(self.total_label_counts, dim=0)
        # [user_id][label] = 样本比例
        self.label_weights = torch.zeros_like(self.label_counts, dtype=torch.float32)
        for user_id in range(self.label_counts.shape[0]):
            for label in range(self.label_counts.shape[1]):
                self.label_weights[user_id][label] = self.label_counts[user_id][label].item() / self.total_label_counts[label].item()
        # 创建客户端
        for i in range(self.total_users):
            user = self.create_client(args, self.client_ids[i], args.model_name)
            self.users.append(user)
        # 创建组
        self.groups = {}
        for user in self.users:
            if user.model_name not in self.groups.keys():
                self.groups[user.model_name] = Group(user.model)
            self.groups[user.model_name].users.append(user)
        # 度量
        self.mertrics = {
            "global_avg_acc": [],
            "global_avg_loss": []}
        print("Finished creating server and clients.")


    def create_client(self, args, client_id, local_model_name):
        return LocalUser(args, self, client_id, local_model_name, self.train_dataset, self.test_dataset, self.train_groups[client_id], self.test_groups[client_id], self.label_counts[client_id])


    def proceed(self):
        self.evaluate(-1)
        for glob_iter in range(self.num_glob_iters):
            print("\n\n------------- Round number: ",glob_iter, " -------------\n")
            self.selected_users, self.user_idxs = self.select_users(glob_iter, self.per_users)
            self.local_train(glob_iter)
            self.evaluate(glob_iter)
        return self.mertrics


    # round = 第几次迭代, num_users = 20 想要选择多少个客户端
    def select_users(self, round, num_users):
        num_users = min(num_users, len(self.users))
        user_idxs = np.random.choice(range(len(self.users)), num_users, replace=False)
        return [self.users[i] for i in user_idxs], user_idxs
        

    def local_train(self, glob_iter):
        print("\n------- Local Training...... \n")
        train_loss_list = []
        for user in self.selected_users:
            loss = user.train(glob_iter)
            train_loss_list.append(loss)
        global_avg_loss = np.mean(train_loss_list)
        self.mertrics["global_avg_loss"].append(global_avg_loss)
        print("loss: %.2f" % (global_avg_loss))


    def evaluate(self, glob_iter):
        print("\n------- Testing...... \n")
        test_acc_list = []
        for user in self.users:
            acc = user.test()
            test_acc_list.append(acc)
        global_avg_acc = np.mean(test_acc_list)
        self.mertrics["global_avg_acc"].append(global_avg_acc)
        print("acc: %.2f" % (global_avg_acc))
