import torch
import torch.nn as nn

import torch.nn.functional as F


class LSTMBaseline(nn.Module):
    def __init__(self, config):
        super(LSTMBaseline, self).__init__()
        dataset = config.dataset
        target_class = config.target_class
        self.is_bidirectional = config.bidirectional
        self.mode = config.mode

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
        self.dropout = nn.Dropout(config.dropout)
        if self.is_bidirectional:
            self.fc1 = nn.Linear(2 * config.hidden_dim, target_class)
        else:
            self.fc1 = nn.Linear(config.hidden_dim, target_class)

    def forward(self, x):
        if self.mode == 'rand':
            x = self.embed(x)
        elif self.mode == 'static':
            x = self.static_embed(x)
        elif self.mode == 'non-static':
            x = self.non_static_embed(x)
        else:
            print("Unsupported Mode")
            exit()
        x, _ = self.lstm(x)
        x = F.relu(torch.transpose(x, 1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.dropout(x)
        logit = self.fc1(x) # (batch, target_size)
        return logit
