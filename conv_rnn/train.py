import argparse
import os
import random

from torch import utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch
import torch.nn as nn

import data
import model

class RandomSearch(object):
    def __init__(self, params):
        self.params = params

    def __iter__(self):
        param_space = list(GridSearch(self.params))
        random.shuffle(param_space)
        for param in param_space:
            yield param

class GridSearch(object):
    def __init__(self, params):
        self.params = params
        self.param_lengths = [len(param) for param in self.params]
        self.indices = [1] * len(params)

    def _update(self, carry_idx):
        if carry_idx >= len(self.params):
            return True
        if self.indices[carry_idx] < self.param_lengths[carry_idx]:
            self.indices[carry_idx] += 1
            return False
        else:
            self.indices[carry_idx] = 1
            return False or self._update(carry_idx + 1)

    def __iter__(self):
        self.stop_next = False
        self.indices = [1] * len(self.params)
        return self

    def __next__(self):
        if self.stop_next:
            raise StopIteration
        result = [param[idx - 1] for param, idx in zip(self.params, self.indices)]
        self.indices[0] += 1
        if self.indices[0] == self.param_lengths[0] + 1:
            self.indices[0] = 1
            self.stop_next = self._update(1)
        return result

def train(**kwargs):
    mbatch_size = kwargs["mbatch_size"]
    n_epochs = kwargs["n_epochs"]
    restore = kwargs["restore"]
    verbose = not kwargs["quiet"]
    lr = kwargs["lr"]
    weight_decay = kwargs["weight_decay"]
    seed = kwargs["seed"]

    if not kwargs["no_cuda"]:
        torch.cuda.set_device(kwargs["gpu_number"])
    model.set_seed(seed)
    embed_loader = data.SSTEmbeddingLoader("data")
    if restore:
        conv_rnn = torch.load(kwargs["input_file"])
    else:
        id_dict, weights, unk_vocab_list = embed_loader.load_embed_data()
        word_model = model.SSTWordEmbeddingModel(id_dict, weights, unk_vocab_list)
        if not kwargs["no_cuda"]:
            word_model.cuda()
        conv_rnn = model.ConvRNNModel(word_model, **kwargs)
        if not kwargs["no_cuda"]:
            conv_rnn.cuda()

    conv_rnn.train()
    criterion = nn.CrossEntropyLoss()
    parameters = list(filter(lambda p: p.requires_grad, conv_rnn.parameters()))
    optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, patience=kwargs["dev_per_epoch"] * 4)
    train_set, dev_set, test_set = data.SSTDataset.load_sst_sets("data")

    collate_fn = conv_rnn.convert_dataset
    train_loader = utils.data.DataLoader(train_set, shuffle=True, batch_size=mbatch_size, drop_last=True, 
        collate_fn=collate_fn)
    dev_loader = utils.data.DataLoader(dev_set, batch_size=len(dev_set), collate_fn=collate_fn)
    test_loader = utils.data.DataLoader(test_set, batch_size=len(test_set), collate_fn=collate_fn)

    def evaluate(loader, dev=True):
        conv_rnn.eval()
        for m_in, m_out in loader:
            scores = conv_rnn(m_in)
            loss = criterion(scores, m_out).cpu().data[0]
            n_correct = (torch.max(scores, 1)[1].view(m_in.size(0)).data == m_out.data).sum()
            accuracy = n_correct / m_in.size(0)
            scheduler.step(accuracy)
            if dev and accuracy >= evaluate.best_dev:
                evaluate.best_dev = accuracy
                print("Saving best model ({})...".format(accuracy))
                torch.save(conv_rnn, kwargs["output_file"])
            if verbose:
                print("{} set accuracy: {}, loss: {}".format("dev" if dev else "test", accuracy, loss))
        conv_rnn.train()
    evaluate.best_dev = 0

    for epoch in range(n_epochs):
        print("Epoch number: {}".format(epoch), end="\r")
        if verbose:
            print()
        i = 0
        for j, (train_in, train_out) in enumerate(train_loader):
            optimizer.zero_grad()

            if not kwargs["no_cuda"]:
                train_in.cuda()
                train_out.cuda()

            scores = conv_rnn(train_in)
            loss = criterion(scores, train_out)
            loss.backward()
            optimizer.step()
            accuracy = (torch.max(scores, 1)[1].view(-1).data == train_out.data).sum() / mbatch_size
            if verbose and i % (mbatch_size * 10) == 0:
                print("accuracy: {}, {} / {}".format(accuracy, j * mbatch_size, len(train_set)))
            i += mbatch_size
            if i % (len(train_set) // kwargs["dev_per_epoch"]) < mbatch_size:
                evaluate(dev_loader)
    evaluate(test_loader, dev=False)
    return evaluate.best_dev

def do_random_search(given_params):
    test_grid = [[0.15, 0.2], [4, 5, 6], [150, 200], [3, 4, 5], [200, 300], [200, 250]]
    max_params = None
    max_acc = 0.
    for args in RandomSearch(test_grid):
        sf, gc, hid, seed, fc_size, fmaps = args
        print("Testing {}".format(args))
        given_params.update(dict(n_epochs=7, quiet=True, gradient_clip=gc, hidden_Size=hid, seed=seed, 
            n_feature_maps=fmaps, fc_size=fc_size))
        dev_acc = train(**given_params)
        print("Dev accuracy: {}".format(dev_acc))
        if dev_acc > max_acc:
            print("Found current max")
            max_acc = dev_acc
            max_params = args
    print("Best params: {}".format(max_params))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_per_epoch", default=9, type=int)
    parser.add_argument("--fc_size", default=200, type=int)
    parser.add_argument("--gpu_number", default=0, type=int)
    parser.add_argument("--hidden_size", default=200, type=int)
    parser.add_argument("--input_file", default="saves/model.pt", type=str)
    parser.add_argument("--lr", default=1E-1, type=float)
    parser.add_argument("--mbatch_size", default=64, type=int)
    parser.add_argument("--n_epochs", default=30, type=int)
    parser.add_argument("--n_feature_maps", default=200, type=float)
    parser.add_argument("--n_labels", default=5, type=int)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--output_file", default="saves/model.pt", type=str)
    parser.add_argument("--random_search", action="store_true", default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--rnn_type", choices=["lstm", "gru"], default="lstm", type=str)
    parser.add_argument("--seed", default=3, type=int)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--weight_decay", default=1E-4, type=float)
    args = parser.parse_args()
    if args.random_search:
        do_random_search(vars(args))
        return
    train(**vars(args))

if __name__ == "__main__":
    main()

