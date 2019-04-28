import torch.nn as nn
from layers import *


class BiLSTM(nn.Module):
    def __init__(self, hidden_dim=128, dropout_rate=0.1, layer_num=1):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.bilstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            layer_num, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.init_weights()

    def init_weights(self):
        for p in self.bilstm.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                p.data.zero_()

    def forward(self, x, input_mask=None):
        output, _ = self.bilstm(x)
        return output


class MultiAttn(nn.Module):
    def __init__(self, hidden_dim=128, dropout_rate=0.1, layer_num=1):
        super(MultiAttn, self).__init__()
        self.attn = nn.ModuleList([MultiHeadAttention(d_k=64, d_v=64, d_model=hidden_dim,
                                                      n_heads=3,
                                                      dropout=dropout_rate)
                                   for i in range(layer_num)])

    def forward(self, x, input_mask=None):
        for layer in self.attn:
            x, _ = layer(x, x, x, input_mask)
        return x


class Conv(nn.Module):
    def __init__(self, hidden_size):
        super(Conv, self).__init__()
        self.layer1 = nn.Sequential(  # input (batch_size, 1, 128, 768)
            nn.Conv2d(1, 2, kernel_size=(2, hidden_size), stride=1, padding=1),
            nn.BatchNorm1d(2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(  # (batch_size, 2, 66, 768)
            nn.Conv2d(2, 1, kernel_size=(3, hidden_size), stride=1, padding=1),  # (batch_size, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)
        out = self.layer2(out)
        return out
