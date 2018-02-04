import argparse
import os

import torch
import torch.utils.data as data

class Configs(object):
    @staticmethod
    def base_config():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="sick", choices=["sick"])
        parser.add_argument("--input_model", type=str, default="local_saves/model.pt")
        parser.add_argument("--lr", type=float, default=1E-4)
        parser.add_argument("--mbatch_size", type=int, default=40)
        parser.add_argument("--output_model", type=str, default="local_saves/model.pt")
        parser.add_argument("--restore", action="store_true", default=False)
        parser.add_argument("--wordvecs_file", type=str, default="local_data/glove/glove.840B.300d.txt")
        return parser.parse_known_args()[0]

    @staticmethod
    def sick_config():
        parser = argparse.ArgumentParser()
        parser.add_argument("--sick_cache", type=str, default="local_data/sick/.vec-cache")
        parser.add_argument("--sick_data", type=str, default="local_data/sick")
        return parser.parse_known_args()[0]

class LabeledEmbeddedDataset(data.Dataset):
    def __init__(self, sentence_indices, labels):
        assert len(sentence_indices) == len(labels)
        self.sentence_indices = sentence_indices
        self.labels = labels

    def __getitem__(self, idx):
        return self.sentence_indices[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

def load_sick(config):
    pass

def load_dataset():
    config = Configs.base_config()
    return _loaders[config.dataset](config)

_loaders = dict(sick=load_sick)