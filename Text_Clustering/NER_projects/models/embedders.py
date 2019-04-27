import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel


class RandomEmbed(nn.Module):
    def __init__(self, config):
        super(RandomEmbed, self).__init__()
        self.dropout_rate = config.hidden_dropout_prob
        self.hidden_size = config.hidden_size
        self.embedding = nn.Embedding(config.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, input_ids, input_mask=None, segment_ids=None):
        embeddings = self.embedding(input_ids)
        return self.dropout(embeddings)


class BertEmbed(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEmbed, self).__init__(config)
        self.dropout_rate = config.hidden_dropout_prob
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, input_mask, segment_ids=None):
        embeddings, _ = self.bert(input_ids, segment_ids, input_mask,
                                  output_all_encoded_layers=False)  # bert_layer: (batch_size, max_seq_len, hidden_size)
        return embeddings
