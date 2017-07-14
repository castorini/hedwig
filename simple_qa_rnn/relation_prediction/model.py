import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        if config.rnn_type.lower() == "gru":
            self.rnn = nn.GRU(input_size=config.d_embed, hidden_size=config.d_hidden,
                               num_layers=config.n_layers, dropout=config.dropout_prob,
                               bidirectional=config.birnn)
        else:
            self.rnn = nn.LSTM(input_size=config.d_embed, hidden_size=config.d_hidden,
                               num_layers=config.n_layers, dropout=config.dropout_prob,
                               bidirectional=config.birnn)


    def forward(self, inputs):
        # shape of `inputs` - (sequence length, batch size, dimension of embedding)
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        if self.config.rnn_type.lower() == "gru":
            h0 = Variable(inputs.data.new(*state_shape).zero_())
            outputs, ht = self.rnn(inputs, h0)
        else:
            h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
            outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1] if not self.config.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class RelationClassifier(nn.Module):

    def __init__(self, config):
        super(RelationClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(p=config.dropout_prob)
        self.relu = nn.ReLU()
        seq_in_size = config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2

        self.out = nn.Sequential(
                        nn.Linear(seq_in_size, seq_in_size), # can apply batch norm after this - add later
                        nn.BatchNorm1d(seq_in_size),
                        self.relu,
                        self.dropout,
                        nn.Linear(seq_in_size, config.d_out)
        )

    def forward(self, batch):
        # shape of `batch` - (sequence length, batch size)
        question_embed = self.embed(batch.question)
        if self.config.fix_emb:
            question_embed = Variable(question_embed.data)
        # shape of `question_embed` - (sequence length, batch size, dimension of embedding)
        question_encoded = self.encoder(question_embed)
        # shape of `question_encoded` - (batch size, number of cells X size of hidden)
        output = self.out(question_encoded)
        scores = F.log_softmax(output)
        return scores
