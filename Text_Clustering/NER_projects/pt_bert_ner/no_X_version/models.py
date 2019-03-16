import torch
import torch.nn as nn
from crf import CRF
from tools import valid_first
from layers import *

class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, rnn_layers=1, dropout_rate=0):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.bilstm = nn.LSTM(
            embedding_dim, hidden_dim // 2,
            rnn_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.init_weights()

    def init_weights(self):
        for p in self.bilstm.parameters():
            if p.dim()>1:
                nn.init.xavier_normal_(p)
            else:
                p.data.zero_()

    def forward(self, embedding):
        outputs, _ = self.bilstm(embedding.transpose(0,1))
        return outputs.transpose(0,1)

class Conv(nn.Module):
    def __init__(self, hidden_size):
        super(Conv, self).__init__()
        self.layer1 = nn.Sequential( # input (batch_size, 1, 128, 768)
            nn.Conv2d(1,2, kernel_size=(2,hidden_size), stride=1, padding=1),
            nn.BatchNorm1d(2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential( #(batch_size, 2, 66, 768)
            nn.Conv2d(2, 1, kernel_size=(3,hidden_size), stride=1, padding=1), #(batch_size, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class SoftmaxDecoder(nn.Module):
    def __init__(self, label_size, input_dim, input_dropout=0.5):
        super(SoftmaxDecoder, self).__init__()
        self.input_dim = input_dim
        self.label_size = label_size
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.linear = torch.nn.Linear(input_dim, label_size)
        self.crit = LabelSmoothing(label_size, 0.1)
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
    def create(cls, label_size, input_dim, input_dropout=0.5):
        return cls(label_size, input_dim, input_dropout)



class CRFDecoder(nn.Module):
    def __init__(self, crf, label_size, input_dim, input_dropout=0.5):
        super(CRFDecoder, self).__init__()
        self.input_dim = input_dim
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.linear = nn.Linear(in_features=input_dim,
                                out_features=label_size)
        self.crf = crf
        self.label_size = label_size

    def forward_model(self, inputs):
        batch_size, seq_len, input_dim = inputs.size()
        output = inputs.contiguous().view(-1, self.input_dim)
        output = self.input_dropout(output)
        output = self.linear.forward(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, predict_mask, labels=None):
        logits = self.forward_model(inputs)
        logits = self.crf.pad_logits(logits)
        p = torch.nn.functional.softmax(logits, -1)  # (batch_size, max_seq_len, num_labels)
        predict_mask, labels, logits = valid_first(predict_mask, labels,logits)
        lens = predict_mask.sum(-1)
        if labels is None:
            #scores, preds = self.crf.viterbi_decode(logits, predict_mask)
            scores, preds = self.crf.viterbi_decode(logits, lens)
            return preds, p
        #return self.score(logits, predict_mask, labels)
        return self.score(logits, lens, labels)

    def score(self, logits, lens, labels):
        norm_score = self.crf.calc_norm_score(logits, lens)
        gold_score = self.crf.calc_gold_score(logits, labels, lens)
        loglik = gold_score - norm_score
        return -loglik.mean()

    @classmethod
    def create(cls, label_size, input_dim, input_dropout=0.2):
        return cls(CRF(label_size + 2), label_size, input_dim, input_dropout)

class AttnSoftmaxDecoder(nn.Module):
    def __init__(self, label_size, input_dim, input_dropout=0.5,
                 key_dim=64, val_dim=64, num_heads=3):
        super(AttnSoftmaxDecoder, self).__init__()
        self.input_dim = input_dim
        self.label_size = label_size
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.linear = nn.Linear(input_dim, label_size)
        self.attn = MultiHeadAttention(key_dim, val_dim, input_dim, num_heads, input_dropout)
        self.crit = LabelSmoothing(label_size, 0.1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear.weight)

    def forward_model(self, inputs, labels_mask=None):
        batch_size, seq_len, input_dim = inputs.size()
        inputs, _ = self.attn(inputs, inputs, inputs, labels_mask)

        output = inputs.contiguous().view(-1, self.input_dim)
        # Fully-connected layer
        output = self.linear.forward(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, predict_mask, label_ids=None):
        logits = self.forward_model(inputs)
        p = torch.nn.functional.softmax(logits, -1)  # (batch_size, max_seq_len, num_labels)
        predict_mask, label_ids, logits = valid_first(predict_mask, label_ids, logits)
        if label_ids is not None:
            # p = torch.nn.functional.softmax(logits, -1)  # (batch_size, max_seq_len, num_labels)
            # one_hot_labels = torch.eye(self.label_size)[label_ids].type_as(p)
            # losses = -torch.log(torch.sum(one_hot_labels * p, -1))  # (batch_size, max_seq_len)
            # masked_losses = torch.masked_select(losses, predict_mask)  # (batch_sum_real_len)
            # return torch.mean(masked_losses)
            return self.crit(logits, label_ids, predict_mask)
        else:
            return torch.argmax(logits, -1), p

    @classmethod
    def create(cls, label_size, input_dim, input_dropout=0.5):
        return cls(label_size, input_dim, input_dropout)


class AttnCRFDecoder(nn.Module):
    def __init__(self,
                 crf, label_size, input_dim, input_dropout=0.5,
                 key_dim=64, val_dim=64, num_heads=3):
        super(AttnCRFDecoder, self).__init__()
        self.input_dim = input_dim
        self.attn1 = MultiHeadAttention(key_dim, val_dim, input_dim, num_heads, input_dropout)
        #self.attn2 = MultiHeadAttention(key_dim, val_dim, input_dim, num_heads, input_dropout)
        self.linear = nn.Linear(input_dim, label_size)
        self.crf = crf
        self.label_size = label_size

    def forward_model(self, inputs, labels_mask=None):
        batch_size, seq_len, input_dim = inputs.size()
        inputs, _ = self.attn1(inputs, inputs, inputs, labels_mask)
        #inputs, _ = self.attn2(inputs, inputs, inputs, labels_mask)

        output = inputs.contiguous().view(-1, self.input_dim)
        # Fully-connected layer
        output = self.linear.forward(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, predict_mask, labels=None):
        logits = self.forward_model(inputs)
        logits = self.crf.pad_logits(logits)
        p = torch.nn.functional.softmax(logits, -1)  # (batch_size, max_seq_len, num_labels)
        predict_mask, labels, logits = valid_first(predict_mask, labels, logits)
        lens = predict_mask.sum(-1)
        if labels is None:
            # scores, preds = self.crf.viterbi_decode(logits, predict_mask)
            scores, preds = self.crf.viterbi_decode(logits, lens)
            return preds, p
        # return self.score(logits, predict_mask, labels)
        return self.score(logits, lens, labels)

    def score(self, logits, lens, labels):
        norm_score = self.crf.calc_norm_score(logits, lens)
        gold_score = self.crf.calc_gold_score(logits, labels, lens)
        loglik = gold_score - norm_score
        return -loglik.mean()

    @classmethod
    def create(cls, label_size, input_dim, input_dropout=0.5, key_dim=64, val_dim=64, num_heads=3):
        return cls(CRF(label_size + 2), label_size, input_dim, input_dropout,
                   key_dim, val_dim, num_heads)


