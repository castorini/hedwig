import os

import torch
from torchtext.data import Field, NestedField, TabularDataset
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors

from datasets.reuters import clean_string, split_sents, process_labels
from datasets.ag_news import process_labels, char_quantize, ALPHABET_DICT


class DBpedia(TabularDataset):
    NAME = 'DBpedia'
    NUM_CLASSES = 14
    IS_MULTILABEL = False
    TEXT_FIELD = Field(batch_first=True, tokenize=clean_string, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True,
                        preprocessing=lambda s: process_labels(s, DBpedia.NUM_CLASSES))

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, path, train=os.path.join('.local_data', 'DBpedia', 'train.csv'),
               test=os.path.join('.local_data', 'DBpedia', 'test.csv'), **kwargs):
        return super(DBpedia, cls).splits(
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


class DBpediaCharQuantized(DBpedia):
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


class DBpediaHierarchical(DBpedia):
    NESTING_FIELD = Field(batch_first=True, tokenize=clean_string)
    TEXT_FIELD = NestedField(NESTING_FIELD, tokenize=split_sents)
