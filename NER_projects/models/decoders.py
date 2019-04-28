import torch
import torch.nn as nn
from crf import CRF
from layers import *
from tools import valid_first


class SoftmaxDecoder(nn.Module):
    def __init__(self, label_size, input_dim, input_dropout, cal_X_loss):
        super(SoftmaxDecoder, self).__init__()
        self.input_dim = input_dim
        self.label_size = label_size
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.linear = torch.nn.Linear(input_dim, label_size)
        self.crit = LabelSmoothing(label_size, 0.1)
        self.cal_X_loss = cal_X_loss
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear.weight)

    def forward_model(self, inputs):
        batch_size, seq_len, input_dim = inputs.size()
        output = inputs.contiguous().view(-1, self.input_dim)
        output = self.input_dropout(output)
        output = self.linear(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, predict_mask, label_ids=None):
        logits = self.forward_model(inputs)
        p = torch.nn.functional.softmax(logits, -1)  # (batch_size, max_seq_len, num_labels)
        if not self.cal_X_loss:
            predict_mask, label_ids, logits = valid_first(predict_mask, label_ids, logits)
        if label_ids is not None:
            # cross entropy loss
            # p = torch.nn.functional.softmax(logits, -1)  # (batch_size, max_seq_len, num_labels)
            # one_hot_labels = torch.eye(self.label_size)[label_ids].type_as(p)
            # losses = -torch.log(torch.sum(one_hot_labels * p, -1))  # (batch_size, max_seq_len)
            # masked_losses = torch.masked_select(losses, predict_mask)  # (batch_sum_real_len)
            # return torch.mean(masked_losses)

            # label smooth with KL-div loss
            return self.crit(logits, label_ids, predict_mask)
        else:
            return torch.argmax(logits, -1), p

    @classmethod
    def create(cls, label_size, input_dim, input_dropout, cal_X_loss):
        return cls(label_size, input_dim, input_dropout, cal_X_loss)


class CRFDecoder(nn.Module):
    def __init__(self, crf, label_size, input_dim, input_dropout, cal_X_loss):
        super(CRFDecoder, self).__init__()
        self.input_dim = input_dim
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.linear = nn.Linear(in_features=input_dim,
                                out_features=label_size)
        self.crf = crf
        self.label_size = label_size
        self.cal_X_loss = cal_X_loss

    def forward_model(self, inputs):
        batch_size, seq_len, input_dim = inputs.size()
        output = inputs.contiguous().view(-1, self.input_dim)
        output = self.input_dropout(output)
        output = self.linear.forward(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, predict_mask, labels=None):
        logits = self.forward_model(inputs)
        p = torch.nn.functional.softmax(logits, -1)  # (batch_size, max_seq_len, num_labels)
        logits = self.crf.pad_logits(logits)
        if not self.cal_X_loss:
            predict_mask, labels, logits = valid_first(predict_mask, labels, logits)
        lens = predict_mask.sum(-1)
        if labels is None:
            scores, preds = self.crf.viterbi_decode(logits, lens)
            return preds, p
        return self.score(logits, lens, labels)

    def score(self, logits, lens, labels):
        norm_score = self.crf.calc_norm_score(logits, lens)
        gold_score = self.crf.calc_gold_score(logits, labels, lens)
        loglik = gold_score - norm_score
        return -loglik.mean()

    @classmethod
    def create(cls, label_size, input_dim, input_dropout, cal_X_loss):
        return cls(CRF(label_size + 2), label_size, input_dim, input_dropout, cal_X_loss)
