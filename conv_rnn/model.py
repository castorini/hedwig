import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_func

import data

class ConvRNNModel(nn.Module):
    def __init__(self, word_model, **config):
        super().__init__()
        embedding_dim = word_model.dim
        self.word_model = word_model
        self.hidden_size = config["hidden_size"]
        fc_size = config["fc_size"]
        self.batch_size = config["mbatch_size"]
        n_fmaps = config["n_feature_maps"]
        self.rnn_type = config["rnn_type"]

        self.h_0_cache = torch.autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_size))
        self.c_0_cache = torch.autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_size))

        self.no_cuda = config["no_cuda"]
        if not self.no_cuda:
            self.h_0_cache = self.h_0_cache.cuda()
            self.c_0_cache = self.c_0_cache.cuda()

        if self.rnn_type.upper() == "LSTM":
            self.bi_rnn = nn.LSTM(embedding_dim, self.hidden_size, 1, batch_first=True, bidirectional=True)
        elif self.rnn_type.upper() == "GRU":
            self.bi_rnn = nn.GRU(embedding_dim, self.hidden_size, 1, batch_first=True, bidirectional=True)
        else:
            raise ValueError("RNN type must be one of LSTM or GRU")
        self.conv = nn.Conv2d(1, n_fmaps, (1, self.hidden_size * 2))
        self.fc1 = nn.Linear(n_fmaps + 2 * self.hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, config["n_labels"])

    def convert_dataset(self, dataset):
        dataset = np.stack(dataset)
        model_in = dataset[:, 1].reshape(-1)
        model_out = dataset[:, 0].flatten().astype(np.int)
        model_out = torch.autograd.Variable(torch.from_numpy(model_out))
        model_in = self.preprocess(model_in)
        model_in = torch.autograd.Variable(model_in)
        if not self.no_cuda:
            model_out = model_out.cuda()
            model_in = model_in.cuda()
        return (model_in, model_out)

    def preprocess(self, sentences):
        return torch.from_numpy(np.array(self.word_model.lookup(sentences)))

    def forward(self, x):
        x = self.word_model(x) # shape: (batch, max sent, embed dim)
        if x.size(0) == self.batch_size:
            h_0 = self.h_0_cache
            c_0 = self.c_0_cache
        else:
            h_0 = torch.autograd.Variable(torch.zeros(2, x.size(0), self.hidden_size))
            c_0 = torch.autograd.Variable(torch.zeros(2, x.size(0), self.hidden_size))
            if not self.no_cuda:
                h_0 = h_0.cuda()
                c_0 = c_0.cuda()
        if self.rnn_type.upper() == "LSTM":
            rnn_seq, rnn_out = self.bi_rnn(x, (h_0, c_0)) # shape: (batch, seq len, 2 * hidden_size), (2, batch, hidden_size)
            rnn_out = rnn_out[0] # (h_0, c_0)
        else:
            rnn_seq, rnn_out = self.bi_rnn(x, h_0) # shape: (batch, 2, hidden_size)
        rnn_out.data = rnn_out.data.permute(1, 0, 2)
        x = self.conv(rnn_seq.unsqueeze(1)).squeeze(3) # shape: (batch, channels, seq len)
        x = nn_func.relu(x) # shape: (batch, channels, seq len)
        x = nn_func.max_pool1d(x, x.size(2)) # shape: (batch, channels)
        out = [t.squeeze(1) for t in rnn_out.chunk(2, 1)]
        out.append(x)
        x = torch.cat(out, 1).squeeze(2)
        x = nn_func.relu(self.fc1(x))
        return self.fc2(x)

class WordEmbeddingModel(nn.Module):
    def __init__(self, id_dict, weights, unknown_vocab=[], static=True, padding_idx=0):
        super().__init__()
        vocab_size = len(id_dict) + len(unknown_vocab)
        self.lookup_table = id_dict
        last_id = max(id_dict.values())
        for word in unknown_vocab:
            last_id += 1
            self.lookup_table[word] = last_id
        self.dim = weights.shape[1]
        self.weights = np.concatenate((weights, np.random.rand(len(unknown_vocab), self.dim) / 2 - 0.25))
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, self.dim, padding_idx=padding_idx)
        self.embedding.weight.data.copy_(torch.from_numpy(self.weights))
        if static:
            self.embedding.weight.requires_grad = False

    @classmethod
    def make_random_model(cls, id_dict, unknown_vocab=[], dim=300):
        weights = np.random.rand(len(id_dict), dim) - 0.5
        return cls(id_dict, weights, unknown_vocab, static=False)

    def forward(self, x):
        return self.embedding(x)

    def lookup(self, sentences):
        raise NotImplementedError

class SSTWordEmbeddingModel(WordEmbeddingModel):
    def __init__(self, id_dict, weights, unknown_vocab=[]):
        super().__init__(id_dict, weights, unknown_vocab, padding_idx=16259)

    def lookup(self, sentences):
        indices_list = []
        max_len = 0
        for sentence in sentences:
            indices = []
            for word in data.sst_tokenize(sentence):
                try:
                    index = self.lookup_table[word]
                    indices.append(index)
                except KeyError:
                    continue
            indices_list.append(indices)
            if len(indices) > max_len:
                max_len = len(indices)
        for indices in indices_list:
            indices.extend([self.padding_idx] * (max_len - len(indices))) 
        return indices_list

def set_seed(seed=0, no_cuda=False):
    np.random.seed(seed)
    if not no_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
