import numpy as np
import torch
from torch.nn import Module
from torch.nn.functional import sigmoid
import copy
from sklearn.metrics import average_precision_score, roc_auc_score

def recon_loss(logits, ground_truths, EPS=1e-10):
    pos_mask = (ground_truths == 1.0)
    pos_loss = -torch.log(sigmoid(logits[pos_mask]) + EPS).mean()
    neg_loss = -torch.log(1.0 - sigmoid(logits[~pos_mask]) + EPS).mean()

    return pos_loss + neg_loss

def hitsk(model, test_data_split, k):
    test_data_split_pos = copy.copy(test_data_split)
    test_data_split_pos.edge_label_index = test_data_split.pos_edge_label_index

    test_data_split_neg = copy.copy(test_data_split)
    test_data_split_neg.edge_label_index = test_data_split.neg_edge_label_index

    return compute_hitsk(model(test_data_split_pos).x, model(test_data_split_neg).x, k)

def compute_hitsk(y_pred_pos, y_pred_neg, k):
    tot = (y_pred_pos > torch.sort(y_pred_neg, descending=True)[0][k]).sum()
    return tot / y_pred_pos.size(0)

def average_precision(model, test_data):
    preds = sigmoid(model(test_data).x).cpu().detach().numpy()
    labels = test_data.edge_label.cpu().detach().numpy()
    return average_precision_score(labels, preds)

def auc_loss(logits, ground_truths):
    return 1.0 - roc_auc_score(ground_truths.cpu(), logits.x.cpu())

def ap_loss(logits, ground_truths):
    preds = sigmoid(logits.x).cpu().detach().numpy()
    labels = ground_truths.cpu().detach().numpy()
    return 1.0 - average_precision_score(labels, preds)

def losses_sum_closure(losses):
    return lambda logits, ground_truths: np.sum([loss(logits, ground_truths) for loss in losses])

class StandardLossWrapper(Module):
    def __init__(self, norm, loss):
        super().__init__()
        self.loss = loss
        self.norm = norm

    def forward(self, batch, ground_truth):
        return self.norm * self.loss(batch.x, ground_truth)

def kl_loss(mu, logstd):
    kld = 1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2)
    return -0.5 * torch.mean(torch.sum(kld, dim=1))

class VGAELossWrapper(Module):
    def __init__(self, norm, loss):
        super().__init__()
        self.loss = loss
        self.norm = norm

    def forward(self, batch, ground_truth):
        loss_val = self.norm * self.loss(batch.x, ground_truth)
        kl_val = (0.5 / batch.x.size(0)) * kl_loss(batch.mu, batch.logstd)
        return loss_val + kl_val
