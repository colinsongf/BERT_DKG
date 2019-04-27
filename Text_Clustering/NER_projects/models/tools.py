import torch

def valid_first(predict_mask, labels=None, logits=None):
    '''
    When not predicting the latter token(padding tokens),
    bring first the main token of logits, predict_mask, labels.
    this method is used when we don't want to add the loss of 'X' tokens

    :param logits: (batch_size, seq_len, num_labels)
    :param predict_mask: (batch_size, seq_len)
    :param labels: (batch_size, seq_len)
    :return: logits, predict_mask, labels in same size
    '''
    max_len = predict_mask.size(1)
    lens = predict_mask.sum(-1).cpu().numpy()# (batch_size)
    if labels is not None:
        flattened_labels = labels[predict_mask==1] # (sum_words_size)
    if logits is not None:
        flattened_logits = logits[predict_mask==1] # (sum_words_size, num_labels)
    flattened_masks = predict_mask[predict_mask==1] # (sum_words_size)

    def padded(v, max_len):
        if max_len-v.size()[0] > 0:
            pad = v.new_zeros([max_len-v.size()[0], *v.size()[1:]], requires_grad=False)
            return torch.cat([v, pad], 0)
        return v

    start_len = 0
    if logits is not None:
        new_logits = logits.clone()
    else:
        new_logits = None
    if labels is not None:
        new_labels = labels.clone()
    else:
        new_labels = None
    new_masks = predict_mask.clone()
    for ix, len in enumerate(lens):
        if logits is not None:
            new_logits[ix] = padded(flattened_logits[start_len:start_len + len, :], max_len)
        if labels is not None:
            new_labels[ix] = padded(flattened_labels[start_len:start_len + len], max_len)
        new_masks[ix] = padded(flattened_masks[start_len:start_len + len], max_len)
        start_len += len
    return new_masks, new_labels, new_logits

# logits = torch.tensor([[[1.2,2.1], [2.8,2.1], [2.2,-2.1]], [[4.1,2.2], [2.8,2.1], [2.2,-2.1]]]) # 2, 3, 2
# predict_mask = torch.tensor([[1,1,1],[1,0,1]]) # 2, 3
# labels = torch.tensor([[1,0,0],[0, 1, 1]]) # 2, 3
# print(valid_first(logits, predict_mask, labels))