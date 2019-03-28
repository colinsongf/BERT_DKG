import torch
import torch.nn as nn
from crf import CRF


class SoftmaxDecoder(nn.Module):
    def __init__(self, label_size, input_dim, input_dropout=0.5):
        super(SoftmaxDecoder, self).__init__()
        self.input_dim = input_dim
        self.label_size = label_size
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.linear = torch.nn.Linear(input_dim, label_size)
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
        if label_ids is not None:
            p = torch.nn.functional.softmax(logits, -1)  # (batch_size, max_seq_len, num_labels)
            one_hot_labels = torch.eye(self.label_size)[label_ids].type_as(p)
            losses = -torch.log(torch.sum(one_hot_labels * p, -1))  # (batch_size, max_seq_len)
            masked_losses = torch.masked_select(losses, predict_mask)  # (batch_sum_real_len)
            return torch.mean(masked_losses)
        else:
            return torch.argmax(logits, -1)

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
        # logits, predict_mask, labels = valid_first(logits, predict_mask, labels)
        lens = predict_mask.sum(-1)
        if labels is None:
            # scores, preds = self.crf.viterbi_decode(logits, predict_mask)
            scores, preds = self.crf.viterbi_decode(logits, lens)
            return preds
        # return self.score(logits, predict_mask, labels)
        return self.score(logits, lens, labels)

    def score(self, logits, predict_mask, labels):
        norm_score = self.crf.calc_norm_score(logits, predict_mask)
        gold_score = self.crf.calc_gold_score(logits, labels, predict_mask)
        loglik = gold_score - norm_score
        return -loglik.mean()

    @classmethod
    def create(cls, label_size, input_dim, input_dropout=0.2):
        return cls(CRF(label_size + 2), label_size, input_dim, input_dropout)


def valid_first(logits, predict_mask, labels=None):
    '''
    when not predicting the latter token, bring first the main token of logits, predict_mask, labels
    :param logits: (batch_size, seq_len, num_labels)
    :param predict_mask: (batch_size, seq_len)
    :param labels: (batch_size, seq_len)
    :return: logits, predict_mask, labels in same size
    '''
    max_len = logits.size(1)
    lens = predict_mask.sum(-1).cpu().numpy()  # (batch_size)
    if labels is not None:
        flattened_labels = labels[predict_mask == 1]  # (sum_words_size)
    flattened_logits = logits[predict_mask == 1]  # (sum_words_size, num_labels)
    flattened_masks = predict_mask[predict_mask == 1]  # (sum_words_size)

    def padded(v, max_len):
        if max_len - v.size()[0] > 0:
            pad = v.new_zeros([max_len - v.size()[0], *v.size()[1:]], requires_grad=False)
            return torch.cat([v, pad], 0)
        return v

    start_len = 0
    new_logits = logits.clone()
    if labels is not None:
        new_labels = labels.clone()
    else:
        new_labels = None
    new_masks = predict_mask.clone()
    for ix, len in enumerate(lens):
        new_logits[ix] = padded(flattened_logits[start_len:start_len + len, :], max_len)
        if labels is not None:
            new_labels[ix] = padded(flattened_labels[start_len:start_len + len], max_len)
        new_masks[ix] = padded(flattened_masks[start_len:start_len + len], max_len)
        start_len += len
    return new_logits, new_masks, new_labels

# logits = torch.tensor([[[1.2,2.1], [2.8,2.1], [2.2,-2.1]], [[4.1,2.2], [2.8,2.1], [2.2,-2.1]]]) # 2, 3, 2
# predict_mask = torch.tensor([[1,1,1],[1,0,1]]) # 2, 3
# labels = torch.tensor([[1,0,0],[0, 1, 1]]) # 2, 3
# print(valid_first(logits, predict_mask, labels))
