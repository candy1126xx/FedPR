
from Local.user import LocalUser
import torch, numpy as np, copy


class AMPUser(LocalUser):
    def __init__(self, args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts):
        super(AMPUser, self).__init__(args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts)
        self.alphaK = 10000
        self.lamda = 1.0
        self.client_u = copy.deepcopy(self.model)


    def _train(self, glob_iter, x, y, model):
        representation, logits, scores = model(x)
        loss = self.predict_loss(scores, y)
        params = self.model.weight_flatten()
        params_ = self.client_u.weight_flatten()
        sub = params - params_
        loss += self.lamda/self.alphaK/2 * torch.dot(sub, sub)
        loss.backward()
        return loss.item()
