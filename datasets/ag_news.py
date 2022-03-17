import os
import sys
import csv

import torch
from torchtext.data import Field, TabularDataset
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors

from datasets.reuters import clean_string

csv.field_size_limit(sys.maxsize)


class AGNews(TabularDataset):
    NAME = 'AG_NEWS'
    NUM_CLASSES = 4
    IS_MULTILABEL = False

    TEXT_FIELD = Field(batch_first=True, tokenize=clean_string, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True,
                        preprocessing=lambda s: AGNews.process_labels(s))

    @classmethod
    def process_labels(cls, label_str):
        label_num = int(label_str)  # label is one of "1", "2", "3", "4"
        label = [0.0] * cls.NUM_CLASSES
        label[label_num - 1] = 1.0
        return label

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, path, train=os.path.join('.ag_news', 'train.csv'),
               test=os.path.join('.ag_news', 'test.csv'), **kwargs):
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



