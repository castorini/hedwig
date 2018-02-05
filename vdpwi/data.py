import argparse
import os

import torch
import torch.nn as nn
import torch.utils.data as data

class Configs(object):
    @staticmethod
    def base_config():
        parser = argparse.ArgumentParser()
        parser.add_argument("--cpu", action="store_true", default=False)
        parser.add_argument("--dataset", type=str, default="sick", choices=["sick"])
        parser.add_argument("--decay", type=float, default=0.95)
        parser.add_argument("--input_model", type=str, default="local_saves/model.pt")
        parser.add_argument("--lr", type=float, default=1E-4)
        parser.add_argument("--mbatch_size", type=int, default=1)
        parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--n_epochs", type=int, default=40)
        parser.add_argument("--n_labels", type=int, default=5)
        parser.add_argument("--output_model", type=str, default="local_saves/model.pt")
        parser.add_argument("--restore", action="store_true", default=False)
        parser.add_argument("--rnn_hidden_dim", type=int, default=250)
        parser.add_argument("--wordvecs_file", type=str, default="local_data/glove/glove.840B.300d.txt")
        return parser.parse_known_args()[0]

    @staticmethod
    def sick_config():
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_labels", type=int, default=5)
        parser.add_argument("--sick_cache", type=str, default="local_data/sick/.vec-cache")
        parser.add_argument("--sick_data", type=str, default="local_data/sick")
        return parser.parse_known_args()[0]

class LabeledEmbeddedDataset(data.Dataset):
    def __init__(self, sentence_indices1, sentence_indices2, labels):
        assert len(sentence_indices1) == len(labels) == len(sentence_indices2)
        self.sentence_indices1 = sentence_indices1
        self.sentence_indices2 = sentence_indices2
        self.labels = labels

    def __getitem__(self, idx):
        return self.sentence_indices1[idx], self.sentence_indices2[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

def load_sick():
    config = Configs.sick_config()
    def fetch_indices(name):
        sentence_indices = []
        filename = os.path.join(config.sick_data, dataset, name)
        with open(filename) as f:
            for line in f:
                indices = [embed_ids.get(word, padding_idx) for word in line.strip().split()]
                sentence_indices.append(indices)
        return sentence_indices

    sets = []
    embeddings = []
    embed_ids = {}
    with open(os.path.join(config.sick_cache)) as f:
        for i, line in enumerate(f):
            word, vec = line.split(" ", 1)
            vec = list(map(float, vec.strip().split()))
            embed_ids[word] = i
            embeddings.append(vec)
    padding_idx = len(embeddings)
    embeddings.append([0.0] * 300)

    for dataset in ("train", "dev", "test"):
        filename = os.path.join(config.sick_data, dataset, "sim_sparse.txt")
        labels = []
        with open(filename) as f:
            for line in f:
                labels.append([float(val) for val in line.split()])
        indices1 = fetch_indices("a.toks")
        indices2 = fetch_indices("b.toks")
        sets.append(LabeledEmbeddedDataset(indices1, indices2, labels))
    embedding = nn.Embedding(len(embeddings), 300)
    embedding.weight.data.copy_(torch.Tensor(embeddings))
    embedding.weight.requires_grad = False
    return embedding, sets

def load_dataset(dataset):
    return _loaders[dataset]()

_loaders = dict(sick=load_sick)
