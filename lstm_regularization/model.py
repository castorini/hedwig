from copy import deepcopy
import torch
import torch.nn as nn

import torch.nn.functional as F

from lstm_regularization.weight_drop import WeightDrop
from lstm_regularization.embed_regularize import embedded_dropout


class LSTMBaseline(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        target_class = config.target_class
        self.is_bidirectional = config.bidirectional
        self.has_bottleneck_layer = config.bottleneck_layer
        self.mode = config.mode
        self.TAR = config.TAR
        self.AR = config.AR
        self.beta_ema = config.beta_ema  ## Temporal averaging
        self.wdrop = config.wdrop ## Weight dropping
        self.embed_droprate = config.embed_droprate ## Embedding dropout 

        input_channel = 1
        if config.mode == 'rand':
            rand_embed_init = torch.Tensor(config.words_num, config.words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif config.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif config.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            print("Unsupported Mode")
            exit()

        self.lstm = nn.LSTM(config.words_dim, config.hidden_dim, dropout=config.dropout, num_layers=config.num_layers,
                            bidirectional=self.is_bidirectional, batch_first=True)
        ## Wdrop
        if self.wdrop:
            self.lstm = WeightDrop(self.lstm, ['weight_hh_l0'], dropout=self.wdrop)
        self.dropout = nn.Dropout(config.dropout)
        if self.has_bottleneck_layer:
            if self.is_bidirectional:
                self.fc1 = nn.Linear(2 * config.hidden_dim, config.hidden_dim)  # Hidden Bottleneck Layer
                self.fc2 = nn.Linear(config.hidden_dim, target_class)
            else:
                self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim//2)   # Hidden Bottleneck Layer
                self.fc2 = nn.Linear(config.hidden_dim//2, target_class)
        else:
            if self.is_bidirectional:
                self.fc1 = nn.Linear(2 * config.hidden_dim, target_class)
            else:
                self.fc1 = nn.Linear(config.hidden_dim, target_class)
        
        if self.beta_ema>0:
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x, lengths=None):
        if self.mode == 'rand':
            x = embedded_dropout(self.embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.embed(x)
        elif self.mode == 'static':
            x = embedded_dropout(self.static_embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.static_embed(x)
        elif self.mode == 'non-static':
            x = embedded_dropout(self.non_static_embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.non_static_embed(x)
        else:
            print("Unsupported Mode")
            exit()
        if lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        rnn_outs, _ = self.lstm(x)
        rnn_outs_temp = rnn_outs
        #rnn_outs,_ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outs, batch_first = True)

        if lengths is not None:
            rnn_outs,_ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outs, batch_first=True)
            rnn_outs_temp, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outs_temp, batch_first=True)

        x = F.relu(torch.transpose(rnn_outs_temp, 1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.dropout(x)
        if self.has_bottleneck_layer:
            x = F.relu(self.fc1(x))
            # x = self.dropout(x)
            if self.TAR or self.AR:
                return self.fc2(x), rnn_outs.permute(1,0,2)
            return self.fc2(x)
        else:
            if self.TAR or self.AR:
                return self.fc1(x), rnn_outs.permute(1,0,2)
            return self.fc1(x)

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1-self.beta_ema)*p.data)
    
    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p/(1-self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p,avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params
