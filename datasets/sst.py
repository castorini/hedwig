import re

import torch
from torchtext.data import Field, TabularDataset
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip().split()


class SST1(TabularDataset):
    NAME = 'sst-1'
    NUM_CLASSES = 5

    TEXT_FIELD = Field(batch_first=True, tokenize=clean_str_sst)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, path, train='stsa.fine.phrases.train', validation='stsa.fine.dev', test='stsa.fine.test', **kwargs):
        return super(SST1, cls).splits(
            path, train=train, validation=validation, test=test,
            format='tsv', fields=[('label', cls.LABEL_FIELD), ('text', cls.TEXT_FIELD)]
        )

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None,
              unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: path to directory containing word vectors file
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, val, test = cls.splits(path)

        cls.TEXT_FIELD.build_vocab(train, val, test, min_freq=2, vectors=vectors)

        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)
