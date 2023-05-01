from FedAvg.user import AvgUser
import copy


class ProxUser(AvgUser):
    def __init__(self, args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts):
        super(ProxUser, self).__init__(args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts)
        self.mu = 1.0
        self.global_model = copy.deepcopy(self.model)


    def _train(self, glob_iter, x, y, model):
        representation, logits, scores = model(x)
        loss = self.predict_loss(scores, y)
        loss.backward()
        for param, w_g in zip(model.parameters(), self.global_model.parameters()):
            param.grad.data += self.mu * (param.data - w_g.data)
        return loss.item()
