import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.lstm = nn.LSTM(input_size=config.d_embedding, hidden_size=config.d_hidden,
                            num_layers=config.n_layers, dropout=config.dropout_prob,
                            bidirectional=config.birnn, batch_first=True)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # axes semantics are (num_layers, batch_size, hidden_dim)
        n_layers = self.config.n_layers * self.config.n_directions
        out = (Variable(torch.zeros(n_layers, self.config.batch_size, self.config.d_hidden)),
                    Variable(torch.zeros(n_layers, self.config.batch_size, self.config.d_hidden)))
        if self.config.cuda:
            out = ( Variable(torch.zeros(n_layers, self.config.batch_size, self.config.d_hidden).cuda()),
                        Variable(torch.zeros(n_layers, self.config.batch_size, self.config.d_hidden).cuda() ))
        return out

    def forward(self, embeds):
        batch_size = embeds.data.size()[0]
        # sequence_length = embeds.data.size()[1]
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # print("ht size: {}".format(ht.size()))
        return self.hidden[0].transpose(0, 1).contiguous().view(batch_size, -1)  # size - (|B|, |K|)


class RelationPredictor(nn.Module):
    def __init__(self, config):
        super(RelationPredictor, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_embedding)
        self.encoder = Encoder(config)

        self.dropout = nn.Dropout(p=config.dropout_prob)
        self.relu = nn.ReLU()

        # linear layers map from hidden state space to label space
        num_in_features = config.n_layers * config.n_directions * config.d_hidden
        self.hidden2label = nn.Sequential (
                        nn.Linear(num_in_features, num_in_features),
                        nn.BatchNorm1d(num_in_features),
                        self.relu,
                        self.dropout,
                        nn.Linear(num_in_features, num_in_features),
                        nn.BatchNorm1d(num_in_features),
                        self.relu,
                        self.dropout,
                        nn.Linear(num_in_features, config.d_out)
                    )

    # batch_input is Variable of size - (|B|, |S|)
    def forward(self, batch_input):
        batch_input_embed = self.embed(batch_input) # size - (|B|, |S|, |D|)
        # size - (|B|, |X|) where |X| = n_layers * n_directions * d_hidden
        encoded = self.encoder(batch_input_embed)
        rel_space = self.hidden2label(encoded) # size - (|B|, |K|)
        scores = F.log_softmax(rel_space)
        return scores
