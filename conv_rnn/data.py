import os
import re

import numpy as np
import torch.utils.data as data

def sst_tokenize(sentence):
    return sentence.split()

class SSTEmbeddingLoader(object):
    def __init__(self, dirname, fmt="stsa.fine.{}", word2vec_file="word2vec.sst-1"):
        self.dirname = dirname
        self.fmt = fmt
        self.word2vec_file = word2vec_file

    def load_embed_data(self):
        weights = []
        id_dict = {}
        unk_vocab_set = set()
        with open(os.path.join(self.dirname, self.word2vec_file)) as f:
            for i, line in enumerate(f.readlines()):
                word, vec = line.replace("\n", "").split(" ", 1)
                vec = np.array([float(v) for v in vec.split(" ")])
                weights.append(vec)
                id_dict[word] = i
        with open(os.path.join(self.dirname, self.fmt.format("phrases.train"))) as f:
            for line in f.readlines():
                for word in sst_tokenize(line):
                    if word not in id_dict and word not in unk_vocab_set:
                        unk_vocab_set.add(word)
        return (id_dict, np.array(weights), list(unk_vocab_set))

class SSTDataset(data.Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

    @classmethod
    def load_sst_sets(cls, dirname, fmt="stsa.fine.{}"):
        set_names = ["phrases.train", "dev", "test"]
        def read_set(name):
            data_set = []
            with open(os.path.join(dirname, fmt.format(name))) as f:
                for line in f.readlines():
                    sentiment, sentence = line.replace("\n", "").split(" ", 1)
                    data_set.append((sentiment, sentence))
            return np.array(data_set)
        return [cls(read_set(name)) for name in set_names]
