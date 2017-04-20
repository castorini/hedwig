import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class BiLSTM(nn.Module):

    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.config = config

        self.lstm = nn.LSTM(input_size=config.d_embedding, hidden_size=config.d_hidden,
                        num_layers=config.n_layers, dropout=config.dropout_prob,
                        bidirectional=config.birnn, batch_first=True)

        # linear layer maps from hidden state space to label space
        self.hidden2label = nn.Linear(config.n_layers*config.n_directions*config.d_hidden, config.d_out)
        self.hidden = self.init_hidden()
        # self.dropout = nn.Dropout(p=config.dropout_prob)
        # self.log_softmax = nn.LogSoftmax()


    def init_hidden(self):
        # axes semantics are (num_layers, batch_size, hidden_dim)
        n_layers = self.config.n_layers * self.config.n_directions
        return (Variable(torch.zeros(n_layers, self.config.batch_size, self.config.d_hidden)),
                        Variable(torch.zeros(n_layers, self.config.batch_size, self.config.d_hidden)))

    # embeds is Variable of size - (|B|, |S|, |D|)
    def forward(self, embeds):
        batch_size = embeds.data.size()[0]
        sequence_length = embeds.data.size()[1]
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # print("ht size: {}".format(ht.size()))
        rel_space = self.hidden2label(self.hidden[0].transpose(0, 1).contiguous().view(batch_size, -1)) # size - (|B|, |K|)
        scores = F.log_softmax(rel_space)
        return scores
