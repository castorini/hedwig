import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import torch_util


class StackBiLSTMMaxout(nn.Module):

    def __init__(self, h_size=[512, 1024, 2048], d=300, mlp_d=1600, dropout_r=0.1, max_l=60, num_classes=3):
        super().__init__()
        
        self.arch = "SSE"
        self.lstm = nn.LSTM(input_size=d, hidden_size=h_size[0],
                            num_layers=1, bidirectional=True)

        self.lstm_1 = nn.LSTM(input_size=(d + h_size[0] * 2), hidden_size=h_size[1],
                              num_layers=1, bidirectional=True)

        self.lstm_2 = nn.LSTM(input_size=(d + (h_size[0] + h_size[1]) * 2), hidden_size=h_size[2],
                              num_layers=1, bidirectional=True)

        self.max_l = max_l
        self.h_size = h_size

        self.mlp_1 = nn.Linear(h_size[2] * 2 * 4, mlp_d)
        self.mlp_2 = nn.Linear(mlp_d, mlp_d)
        self.sm = nn.Linear(mlp_d, num_classes)

        self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(dropout_r),
                                          self.mlp_2, nn.ReLU(), nn.Dropout(dropout_r),
                                          self.sm])

    def display(self):
        for param in self.parameters():
            print(param.data.size())

    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None):
        sent1 = sent1.permute(2, 0, 1) # from [B * D * T] to [T * B * D]
        sent2 = sent2.permute(2, 0, 1)
        sent1_lengths = torch.tensor([len(s.split(" ")) for s in raw_sent1])
        sent2_lengths = torch.tensor([len(s.split(" ")) for s in raw_sent2])
        if self.max_l:
            sent1_lengths = sent1_lengths.clamp(max=self.max_l)
            sent2_lengths = sent2_lengths.clamp(max=self.max_l)
            if sent1.size(0) > self.max_l:
                sent1 = sent1[:self.max_l, :]
            if sent2.size(0) > self.max_l:
                sent2 = sent2[:self.max_l, :]
        #p_sent1 = self.Embd(sent1)
        #p_sent2 = self.Embd(sent2)
        sent1_layer1_out = torch_util.auto_rnn_bilstm(self.lstm, sent1, sent1_lengths)
        sent2_layer1_out = torch_util.auto_rnn_bilstm(self.lstm, sent2, sent2_lengths)

        # Length truncate
        len1 = sent1_layer1_out.size(0)
        len2 = sent2_layer1_out.size(0)
        p_sent1 = sent1[:len1, :, :] # [T, B, D]
        p_sent2 = sent2[:len2, :, :] # [T, B, D]

        # Using residual connection
        sent1_layer2_in = torch.cat([p_sent1, sent1_layer1_out], dim=2)
        sent2_layer2_in = torch.cat([p_sent2, sent2_layer1_out], dim=2)

        sent1_layer2_out = torch_util.auto_rnn_bilstm(self.lstm_1, sent1_layer2_in, sent1_lengths)
        sent2_layer2_out = torch_util.auto_rnn_bilstm(self.lstm_1, sent2_layer2_in, sent2_lengths)

        sent1_layer3_in = torch.cat([p_sent1, sent1_layer1_out, sent1_layer2_out], dim=2)
        sent2_layer3_in = torch.cat([p_sent2, sent2_layer1_out, sent2_layer2_out], dim=2)

        sent1_layer3_out = torch_util.auto_rnn_bilstm(self.lstm_2, sent1_layer3_in, sent1_lengths)
        sent2_layer3_out = torch_util.auto_rnn_bilstm(self.lstm_2, sent2_layer3_in, sent2_lengths)

        sent1_layer3_maxout = torch_util.max_along_time(sent1_layer3_out, sent1_lengths)
        sent2_layer3_maxout = torch_util.max_along_time(sent2_layer3_out, sent2_lengths)

        # Only use the last layer
        features = torch.cat([sent1_layer3_maxout, sent2_layer3_maxout,
                              torch.abs(sent1_layer3_maxout - sent2_layer3_maxout),
                              sent1_layer3_maxout * sent2_layer3_maxout],
                             dim=1)

        out = self.classifier(features)
        out = F.log_softmax(out, dim=1)
        return out
