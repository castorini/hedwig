import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import os

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class QAModel(nn.Module):
    
    @staticmethod
    def save(model, out_folder, model_fname):
        torch.save(model, os.path.join(out_folder, model_fname))

    @staticmethod
    def load(in_folder, model_fname):
        return torch.load(os.path.join(in_folder, model_fname))

    def __init__(self, input_n_dim, filter_width, conv_filters=100, no_ext_feats=False, ext_feats_size=4, n_classes=2):
        super(QAModel, self).__init__()

        self.no_ext_feats = no_ext_feats

        self.conv_channels = conv_filters
        n_hidden = 2*self.conv_channels + 1

        self.conv_q = nn.Sequential(
            nn.Conv1d(input_n_dim, self.conv_channels, filter_width, padding=filter_width-1),
            nn.Tanh()            
        )

        self.conv_a = nn.Sequential(
            nn.Conv1d(input_n_dim, self.conv_channels, filter_width, padding=filter_width-1),
            nn.Tanh()            
        )
        
        self.combined_feature_vector = nn.Linear(2*self.conv_channels + (0 if no_ext_feats else ext_feats_size), n_hidden)
        #TODO: add +1 to Linear layer^. Will need change in forward function
        self.combined_features_activation = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.hidden = nn.Linear(n_hidden, n_classes)
        self.logsoftmax = nn.LogSoftmax()
                                

    def forward(self, question, answer, ext_feats):                        
        
        q = self.conv_q.forward(question)        
        q = F.max_pool1d(q, q.size()[2])
        q = q.view(-1, self.conv_channels)
        # logger.debug('forward q: {}'.format(q))

        a = self.conv_a.forward(answer)
        a = F.max_pool1d(a, a.size()[2])
        a = a.view(-1, self.conv_channels)

        x = None
        if self.no_ext_feats:
            x = torch.cat([q, a], 1)        
            # logger.debug('no_ext_feats')
        else:
            x = torch.cat([q, a, ext_feats], 1)        
            # logger.debug('with ext_feats')
        
        # logger.debug('featvec x: {}'.format(x))
        # logger.debug(x.creator)

        x = self.combined_feature_vector.forward(x)
        x = self.combined_features_activation.forward(x)
        x = self.dropout(x)
        x = self.hidden(x)
        x = self.logsoftmax(x)

        return x


