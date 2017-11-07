import os

import torch
from torchtext.data.example import Example
from torchtext.data.field import Field
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors

from datasets.castor_dataset import CastorPairDataset
from datasets.idf_utils import get_pairwise_word_to_doc_freq, get_pairwise_overlap_features


class WikiQA(CastorPairDataset):
    NAME = 'wikiqa'
    NUM_CLASSES = 2
    ID_FIELD = Field(sequential=False, tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True)
    TEXT_FIELD = Field(batch_first=True, tokenize=lambda x: x)  # tokenizer is identity since we already tokenized it to compute external features
    EXT_FEATS_FIELD = Field(tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True, tokenize=lambda x: x)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)

    @staticmethod
    def sort_key(ex):
        return len(ex.sentence_1)

    def __init__(self, path):
        """
        Create a WIKIQA dataset instance
        """
        super(WikiQA, self).__init__(path)

    @classmethod
    def splits(cls, path, train='train', validation='dev', test='test', **kwargs):
        return super(WikiQA, cls).splits(path, train=train, validation=validation, test=test, **kwargs)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None, unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: directory containing word vectors file
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, validation, test = cls.splits(path)

        cls.TEXT_FIELD.build_vocab(train, validation, test, vectors=vectors)

        return BucketIterator.splits((train, validation, test), batch_size=batch_size, repeat=False, shuffle=shuffle, device=device)
