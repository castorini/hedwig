import os
import re

import numpy as np
import torch
from torchtext.data import NestedField, Field, TabularDataset
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors

from datasets.reuters import clean_string, clean_string_fl, split_sents


def char_quantize(string, max_length=500):
    identity = np.identity(len(IMDBCharQuantized.ALPHABET))
    quantized_string = np.array([identity[IMDBCharQuantized.ALPHABET[char]] for char in list(string.lower()) if char in IMDBCharQuantized.ALPHABET], dtype=np.float32)
    if len(quantized_string) > max_length:
        return quantized_string[:max_length]
    else:
        return np.concatenate((quantized_string, np.zeros((max_length - len(quantized_string), len(IMDBCharQuantized.ALPHABET)), dtype=np.float32)))


def process_labels(string):
    """
    Returns the label string as a list of integers
    :param string:
    :return:
    """
    return [float(x) for x in string]


class IMDB(TabularDataset):
    NAME = 'IMDB'
    NUM_CLASSES = 10
    TEXT_FIELD = Field(batch_first=True, tokenize=clean_string, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_labels)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, path, train=os.path.join('IMDB', 'data', 'imdb_train.tsv'),
               validation=os.path.join('IMDB', 'data', 'imdb_validation.tsv'),
               test=os.path.join('IMDB', 'data', 'imdb_test.tsv'), **kwargs):
        return super(IMDB, cls).splits(
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
        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)
        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)


class IMDBCharQuantized(IMDB):
    ALPHABET = dict(map(lambda t: (t[1], t[0]), enumerate(list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"""))))
    TEXT_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=char_quantize)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None,
              unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param batch_size: batch size
        :param device: GPU device
        :return:
        """
        train, val, test = cls.splits(path)
        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle, device=device)


class IMDBHierarchical(IMDB):
    NESTING_FIELD = Field(batch_first=True, tokenize=clean_string)
    TEXT_FIELD = NestedField(NESTING_FIELD, tokenize=split_sents)
