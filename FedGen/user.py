import numpy as np, torch, torch.nn as nn
from Local.user import LocalUser


class GenUser(LocalUser):
    def __init__(self, args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts):
        super(GenUser, self).__init__(args, server, id, model, train_dataset, test_dataset, train_idx, test_idx, label_counts)
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean").to(self.device)

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def _train(self, glob_iter, x, y, model):
        self.server.gen_model.eval()
        representations, logits, scores = model(x)
        predict_loss = self.predict_loss(scores, y)
        # 样本 y -> gen_rep -> user_logits = 样本 x -> user_logits
        gen_alpha = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=1)
        y_gen_output = self.server.gen_model(y)['output'].to(self.device)
        y_logit_gen, _ = self.model.forward_rep(y_gen_output)
        y_logit_gen = y_logit_gen.clone().detach()
        user_latent_loss = gen_alpha * self.ensemble_loss(logits, y_logit_gen)
        # 随机 y -> gen_rep -> user_scores = 随机 y
        gen_beta = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=10)
        sampled_y = torch.tensor(np.random.choice(self.available_labels, self.batch_size))
        sampled_y_gen_output = self.server.gen_model(sampled_y)['output'].to(self.device)
        _, scores = self.model.forward_rep(sampled_y_gen_output)
        teacher_loss =  gen_beta * torch.mean(self.server.gen_model.crossentropy_loss(scores, sampled_y.to(self.device)))
        loss = predict_loss + teacher_loss
        loss.backward()
        return loss.item()
