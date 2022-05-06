import os
import sys
import csv
import functools
import numpy as np
import re

import torch
from torchtext.data import Field, NestedField, TabularDataset
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors
from datasets.reuters import split_sents

csv.field_size_limit(sys.maxsize)


def clean_string(string):
    """
    Performs tokenization and string cleaning for the AG_NEWS dataset
    """
    # " #39;" is apostrophe
    string = re.sub(r" #39;", "'", string)
    # " #145;" and " #146;" are left and right single quotes
    string = re.sub(r" #14[56];", "'", string)
    # " #147;" and " #148;" are left and right double quotes
    string = re.sub(r" #14[78];", "\"\"", string)
    # " &lt;" and " &gt;" are < and >
    string = re.sub(r" &lt;", "<", string)
    string = re.sub(r" &gt;", ">", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip().split()


def process_labels(label_str, num_classes):
    label_num = int(label_str)  # label is one of "1", "2", "3", "4"
    label = [0.0] * num_classes
    label[label_num - 1] = 1.0
    return label


class AGNews(TabularDataset):
    NAME = 'AG_NEWS'
    NUM_CLASSES = 4
    IS_MULTILABEL = False

    TEXT_FIELD = Field(batch_first=True, tokenize=clean_string, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True,
                        preprocessing=lambda s: process_labels(s, AGNews.NUM_CLASSES))

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, path, train=os.path.join('.local_data', 'AG_NEWS', 'train.csv'),
               test=os.path.join('.local_data', 'AG_NEWS', 'test.csv'), **kwargs):
        return super(AGNews, cls).splits(
            path, train=train, test=test, format='csv', fields=[('label', cls.LABEL_FIELD), ('text', cls.TEXT_FIELD)]
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

        train, test = cls.splits(path)
        cls.TEXT_FIELD.build_vocab(train, test, vectors=vectors)
        return BucketIterator.splits((train, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)


# Methods and constants common to several datasets
def char_quantize(string, max_length=1000):
    ALPHABET = dict(map(lambda t: (t[1], t[0]), enumerate(list(
        """abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"""))))
    identity = np.identity(len(ALPHABET))
    quantized_string = np.array([identity[ALPHABET[char]] for char in list(string.lower()) if char in ALPHABET], dtype=np.float32)
    if len(quantized_string) > max_length:
        return quantized_string[:max_length]
    else:
        return np.concatenate((quantized_string, np.zeros((max_length - len(quantized_string), len(ALPHABET)), dtype=np.float32)))


ALPHABET_DICT = dict(map(lambda t: (t[1], t[0]), enumerate(list(
    """abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"""))))


class AGNewsCharQuantized(AGNews):
    ALPHABET = ALPHABET_DICT
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
        train, test = cls.splits(path)
        return BucketIterator.splits((train, test), batch_size=batch_size, repeat=False, shuffle=shuffle, device=device)


class AGNewsHierarchical(AGNews):
    NESTING_FIELD = Field(batch_first=True, tokenize=clean_string)
    TEXT_FIELD = NestedField(NESTING_FIELD, tokenize=split_sents)
