import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F

class SentLevelRNN(nn.Module):
    def __init__(self, config):
        super(SentLevelRNN, self).__init__()
        dataset = config.dataset
        sentence_num_hidden = config.sentence_num_hidden
        word_num_hidden = config.word_num_hidden
        target_class = config.target_class
        self.sentence_context_wghts = nn.Parameter(torch.rand(2*sentence_num_hidden, 1))
        self.sentence_context_wghts.data.uniform_(-0.1, 0.1)
        self.sentence_GRU = nn.GRU(2*word_num_hidden, sentence_num_hidden, bidirectional = True)
        self.sentence_linear = nn.Linear(2*sentence_num_hidden, 2*sentence_num_hidden, bias = True)
        self.fc = nn.Linear(2*sentence_num_hidden , target_class)
        self.soft_sent = nn.Softmax()
        self.final_log_soft = F.log_softmax

    def forward(self,x):
            sentence_h,_ = self.sentence_GRU(x)
            x = torch.tanh(self.sentence_linear(sentence_h))
            x = torch.matmul(x, self.sentence_context_wghts)
            x = x.squeeze()
            x = self.soft_sent(x.transpose(1,0))
            x = torch.mul(sentence_h.permute(2,0,1), x.transpose(1,0))
            x = torch.sum(x,dim = 1).transpose(1,0).unsqueeze(0)
            #x = self.final_log_soft(self.fc(x.squeeze(0)))
            x = self.fc(x.squeeze(0))
            return x
