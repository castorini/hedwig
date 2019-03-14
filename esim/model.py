import sys
import math
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def ortho_weight(ndim):
    """
    Random orthogonal weights
    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype('float32')

class LSTM_Cell(nn.Module):

    def __init__(self, device, in_dim, mem_dim):
        super(LSTM_Cell, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        def new_gate():
            h = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
            h.weight.data.copy_(torch.from_numpy(ortho_weight(self.mem_dim)))
            return h

        def new_W():
            w = nn.Linear(self.in_dim, self.mem_dim)
            w.weight.data.copy_(torch.from_numpy(ortho_weight(self.mem_dim)))
            return w

        self.ih = new_gate()
        self.fh = new_gate()
        self.oh = new_gate()
        self.ch = new_gate()

        self.cx = new_W()
        self.ox = new_W()
        self.fx = new_W()
        self.ix = new_W()


    def forward(self, input, h, c):
        u = F.tanh(self.cx(input) + self.ch(h))
        i = F.sigmoid(self.ix(input) + self.ih(h))
        f = F.sigmoid(self.fx(input) + self.fh(h))
        c = i*u + f*c
        o = F.sigmoid(self.ox(input) + self.oh(h))
        h = o * F.tanh(c)
        return c, h

class LSTM(nn.Module):
    def __init__(self, device, in_dim, mem_dim):
        super(LSTM, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.TreeCell = LSTM_Cell(device, in_dim, mem_dim)
        self.output_module = None

    def forward(self, x, x_mask):
        """
        :param x: #step x #sample x dim_emb
        :param x_mask: #step x #sample
        :param x_left_mask: #step x #sample x #step
        :param x_right_mask: #step x #sample x #step
        :return:
        """
        h = Variable(torch.zeros(x.size(1), x.size(2)))
        c = Variable(torch.zeros(x.size(1), x.size(2)))
        if torch.cuda.is_available():
            h=h.to(self.device)
            c=c.to(self.device)
        all_hidden=[]
        for step in range(x.size(0)):
            input=x[step] # #sample x dim_emb
            step_c, step_h=self.TreeCell(input, h, c)
            h=x_mask[step][:,None] * step_h + (1. - x_mask[step])[:,None] * h
            c = x_mask[step][:, None] * step_c + (1. - x_mask[step])[:, None] * c
            all_hidden.append(torch.unsqueeze(h,0))
        return torch.cat(all_hidden,0)

class ESIM(nn.Module):
    """
        Implementation of the multi feed forward network model described in
        the paper "A Decomposable Attention Model for Natural Language
        Inference" by Parikh et al., 2016.
        It applies feedforward MLPs to combinations of parts of the two sentences,
        without any recurrent structure.
    """
    def __init__(self, num_units, num_classes, embedding_size, dropout, device=0,
                 training=True, project_input=True,
                 use_intra_attention=False, distance_biases=10, max_sentence_length=30):
        """
        Create the model based on MLP networks.
        :param num_units: size of the networks
        :param num_classes: number of classes in the problem
        :param embedding_size: size of each word embedding
        :param use_intra_attention: whether to use intra-attention model
        :param training: whether to create training tensors (optimizer)
        :param project_input: whether to project input embeddings to a
            different dimensionality
        :param distance_biases: number of different distances with biases used
            in the intra-attention model
        """
        super(ESIM, self).__init__()
        self.arch = "ESIM"
        self.num_units = num_units
        self.num_classes = num_classes
        self.project_input = project_input
        self.embedding_size=embedding_size
        self.distance_biases=distance_biases
        self.max_sentence_length=max_sentence_length
        self.device = device
        self.dropout = nn.Dropout(p=dropout)

        self.lstm_intra=LSTM(device, embedding_size, num_units)

        self.linear_layer_compare = nn.Sequential(nn.Linear(4*num_units*2, num_units), nn.ReLU(), nn.Dropout(p=dropout))
        #                                          nn.Dropout(p=0.2), nn.Linear(num_units, num_units), nn.ReLU())

        self.lstm_compare=LSTM(device, embedding_size, num_units)

        self.linear_layer_aggregate = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(4*num_units*2, num_units), nn.ReLU(),
                                                    nn.Dropout(p=dropout), nn.Linear(num_units, num_classes))

        self.init_weight()

    def ortho_weight(self):
        """
        Random orthogonal weights
        Used by norm_weights(below), in which case, we
        are ensuring that the rows are orthogonal
        (i.e W = U \Sigma V, U has the same
        # of rows, V has the same # of cols)
        """
        ndim=self.num_units
        W = np.random.randn(ndim, ndim)
        u, s, v = np.linalg.svd(W)
        return u.astype('float32')

    def initialize_lstm(self):
        if torch.cuda.is_available():
            init=torch.Tensor(np.concatenate([self.ortho_weight(),self.ortho_weight(),self.ortho_weight(),self.ortho_weight()], 0)).to(self.device)
        else:
            init = torch.Tensor(
                np.concatenate([self.ortho_weight(), self.ortho_weight(), self.ortho_weight(), self.ortho_weight()], 0))
        return init

    def init_weight(self):
        #nn.init.normal(self.linear_layer_project,mean=0,std=0.1)
        #print(self.linear_layer_attend[3])
        #self.linear_layer_attend[1].weight.data.normal_(0, 0.01)
        #self.linear_layer_attend[1].bias.data.fill_(0)
        #self.linear_layer_attend[4].weight.data.normal_(0, 0.01)
        #self.linear_layer_attend[4].bias.data.fill_(0)
        self.linear_layer_compare[0].weight.data.normal_(0, 0.01)
        self.linear_layer_compare[0].bias.data.fill_(0)
        #self.linear_layer_compare[4].weight.data.normal_(0, 0.01)
        #self.linear_layer_compare[4].bias.data.fill_(0)
        self.linear_layer_aggregate[1].weight.data.normal_(0, 0.01)
        self.linear_layer_aggregate[1].bias.data.fill_(0)
        self.linear_layer_aggregate[4].weight.data.normal_(0, 0.01)
        self.linear_layer_aggregate[4].bias.data.fill_(0)

    def attention_softmax3d(self,raw_attentions):
        reshaped_attentions = raw_attentions.view(-1, raw_attentions.size(2))
        out=nn.functional.softmax(reshaped_attentions, dim=1)
        return out.view(raw_attentions.size(0),raw_attentions.size(1),raw_attentions.size(2))

    def _transformation_input(self,embed_sent, x1_mask):
        embed_sent = self.word_embedding(embed_sent)
        embed_sent = self.dropout(embed_sent)
        hidden=self.lstm_intra(embed_sent, x1_mask)
        return hidden


    def aggregate(self,v1, v2):
        """
        Aggregate the representations induced from both sentences and their
        representations
        :param v1: tensor with shape (batch, time_steps, num_units)
        :param v2: tensor with shape (batch, time_steps, num_units)
        :return: logits over classes, shape (batch, num_classes)
        """
        v1_mean = torch.mean(v1, 0)
        v2_mean = torch.mean(v2, 0)
        v1_max, _ = torch.max(v1, 0)
        v2_max, _ = torch.max(v2, 0)
        out = self.linear_layer_aggregate(torch.cat((v1_mean, v1_max, v2_mean, v2_max), 1))

        #v1_sum=torch.sum(v1,1)
        #v2_sum=torch.sum(v2,1)
        #out=self.linear_layer_aggregate(torch.cat([v1_sum,v2_sum],1))

        return out

    def cosine_interaction(self, tensor1, tensor2):
        """
        :param tensor1: #step1 * dim
        :param tensor2: #step2 * dim
        :return: #step1 * #step2
        """
        simCube_0=tensor1[0].view(1,-1)
        simCube_1=tensor2[0].view(1,-1)
        for i in range(tensor1.size(0)):
            for j in range(tensor2.size(0)):
                if not(i==0 and j==0):
                    simCube_0=torch.cat((simCube_0, tensor1[i].view(1,-1)))
                    simCube_1=torch.cat((simCube_1, tensor2[j].view(1,-1)))
        simCube=F.cosine_similarity(simCube_0, simCube_1)
        return simCube.view(tensor1.size(0),tensor2.size(0))
    
    def create_mask(self, sent):
        masks = []
        sent_lengths = [len(s.split(" ")) for s in sent]
        max_len = max(sent_lengths)

        for s_length in sent_lengths:
            pad_mask = np.zeros(max_len)
            pad_mask[:s_length] = 1
            masks.append(pad_mask)

        masks = np.array(masks)
        return torch.from_numpy(masks).float().to(self.device)

    #def forward(self, x1, x1_mask, x2, x2_mask):
    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None, visualize=False):
        # idx = [i for i in range(embed_sent.size(1) - 1, -1, -1)]
        # if torch.cuda.is_available():
        #   idx = torch.cuda.LongTensor(idx)
        # else:
        #   idx = torch.LongTensor(idx)
        sent1 = sent1.permute(2, 0, 1) # from [B * D * T] to [T * B * D]
        sent2 = sent2.permute(2, 0, 1)
        x1_mask = self.create_mask(raw_sent1)
        x2_mask = self.create_mask(raw_sent2)
        x1_mask = x1_mask.permute(1, 0)
        x2_mask = x2_mask.permute(1, 0)
        #x1 = self.word_embedding(x1)
        x1 = self.dropout(sent1)
        #x2 = self.word_embedding(x2)
        x2 = self.dropout(sent2)
        idx_1 = [i for i in range(x1.size(0) - 1, -1, -1)]
        idx_1 = Variable(torch.LongTensor(idx_1))
        if torch.cuda.is_available():
            idx_1 = idx_1.to(self.device)
        x1_r=torch.index_select(x1,0,idx_1)
        x1_mask_r=torch.index_select(x1_mask,0,idx_1)
        idx_2=[i for i in range(x2.size(0) -1, -1, -1)]
        idx_2 = Variable(torch.LongTensor(idx_2))
        if torch.cuda.is_available():
            idx_2 = Variable(torch.LongTensor(idx_2)).to(self.device)
        x2_r=torch.index_select(x2,0,idx_2)
        x2_mask_r=torch.index_select(x2_mask, 0, idx_2)

        proj1=self.lstm_intra(x1, x1_mask)
        proj1_r=self.lstm_intra(x1_r, x1_mask_r)
        proj2=self.lstm_intra(x2, x2_mask)
        proj2_r=self.lstm_intra(x2_r, x2_mask_r)

        ctx1=torch.cat((proj1, torch.index_select(proj1_r,0,idx_1)),2)
        ctx2=torch.cat((proj2, torch.index_select(proj2_r, 0, idx_2)),2)
        # ctx1: #step1 x #sample x #dimctx
        # ctx2: #step2 x #sample x #dimctx
        ctx1 = ctx1 * x1_mask[:, :, None]
        ctx2 = ctx2 * x2_mask[:, :, None]

        # weight_matrix: #sample x #step1 x #step2
        weight_matrix = torch.matmul(ctx1.permute(1, 0, 2), ctx2.permute(1, 2, 0))
        if visualize:
            return weight_matrix
        weight_matrix_1 = torch.exp(weight_matrix - weight_matrix.max(1, keepdim=True)[0]).permute(1, 2, 0)
        weight_matrix_2 = torch.exp(weight_matrix - weight_matrix.max(2, keepdim=True)[0]).permute(1, 2, 0)

        # weight_matrix_1: #step1 x #step2 x #sample
        weight_matrix_1 = weight_matrix_1 * x1_mask[:, None, :]
        weight_matrix_2 = weight_matrix_2 * x2_mask[None, :, :]

        alpha = weight_matrix_1 / weight_matrix_1.sum(0, keepdim=True)
        beta = weight_matrix_2 / weight_matrix_2.sum(1, keepdim=True)

        self.alpha=alpha
        self.beta=beta

        ctx2_ = (torch.unsqueeze(ctx1,1) * torch.unsqueeze(alpha,3)).sum(0)
        ctx1_ = (torch.unsqueeze(ctx2, 0) * torch.unsqueeze(beta,3)).sum(1)

        # cosine distance and Euclidean distance
        '''
        tmp_result=[]
        for batch_i in range(ctx1.size(1)):
            tmp_result.append(torch.unsqueeze(self.cosine_interaction(ctx1[:,batch_i,:], ctx2[:,batch_i,:]), 0))
        weight_matrix=torch.cat(tmp_result)
        weight_matrix_1 = torch.exp(weight_matrix - weight_matrix.max(1, keepdim=True)[0]).permute(1, 2, 0)
        weight_matrix_2 = torch.exp(weight_matrix - weight_matrix.max(2, keepdim=True)[0]).permute(1, 2, 0)
        # weight_matrix_1: #step1 x #step2 x #sample
        weight_matrix_1 = weight_matrix_1 * x1_mask[:, None, :]
        weight_matrix_2 = weight_matrix_2 * x2_mask[None, :, :]
        alpha = weight_matrix_1 / weight_matrix_1.sum(0, keepdim=True)
        beta = weight_matrix_2 / weight_matrix_2.sum(1, keepdim=True)
        ctx2_cos_ = (torch.unsqueeze(ctx1, 1) * torch.unsqueeze(alpha, 3)).sum(0)
        ctx1_cos_ = (torch.unsqueeze(ctx2, 0) * torch.unsqueeze(beta, 3)).sum(1)
        '''

        inp1 = torch.cat([ctx1, ctx1_, ctx1 * ctx1_, ctx1 - ctx1_], 2)
        inp2 = torch.cat([ctx2, ctx2_, ctx2 * ctx2_, ctx2 - ctx2_], 2)
        #inp1 = torch.cat([ctx1, ctx1_, ctx1_cos_, ctx1 * ctx1_, ctx1 * ctx1_cos_, ctx1 - ctx1_, ctx1 - ctx1_cos_], 2)
        #inp2 = torch.cat([ctx2, ctx2_, ctx2_cos_, ctx2 * ctx2_, ctx2 * ctx2_cos_, ctx2 - ctx2_, ctx2 - ctx2_cos_], 2)
        inp1=self.dropout(self.linear_layer_compare(inp1))
        inp2=self.dropout(self.linear_layer_compare(inp2))
        inp1_r=torch.index_select(inp1, 0, idx_1)
        inp2_r=torch.index_select(inp2, 0, idx_2)

        v1=self.lstm_compare(inp1, x1_mask)
        v2=self.lstm_compare(inp2, x2_mask)
        v1_r = self.lstm_compare(inp1_r, x1_mask)
        v2_r = self.lstm_compare(inp2_r, x2_mask)
        v1=torch.cat((v1, torch.index_select(v1_r, 0, idx_1)),2)
        v2=torch.cat((v2, torch.index_select(v2_r, 0, idx_2)),2)
        out = self.aggregate(v1, v2)
        out = F.log_softmax(out, dim=1)
        return out
