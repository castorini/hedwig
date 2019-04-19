import csv
import os
import sys

import torch
from torchtext.data import NestedField, Field, TabularDataset
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors

from datasets.robust45 import clean_string, split_sents, process_docids, process_labels

csv.field_size_limit(sys.maxsize)


class Robust05(TabularDataset):
    NAME = 'Robust05'
    NUM_CLASSES = 2
    TEXT_FIELD = Field(batch_first=True, tokenize=clean_string, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_labels)
    DOCID_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_docids)
    TOPICS = ['307', '310', '325', '330', '336', '341', '344', '345', '347', '353', '354', '362', '363', '367', '372',
              '375', '378', '389', '393', '394', '397', '399', '404', '408', '416', '419', '426', '427', '433', '435',
              '436', '439', '443']

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, path, train, validation, test, **kwargs):
        return super(Robust05, cls).splits(
            path, train=train, validation=validation, test=test,
            format='tsv', fields=[('label', cls.LABEL_FIELD), ('docid', cls.DOCID_FIELD), ('text', cls.TEXT_FIELD)]
        )

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, topic, batch_size=64, shuffle=True, device=0,
              vectors=None, unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: path to directory containing word vectors file
        :param topic: topic from which articles should be fetched
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train_path = os.path.join('TREC', 'robust05_train_%s.tsv' % topic)
        dev_path = os.path.join('TREC', 'robust05_dev_%s.tsv' % topic)
        test_path = os.path.join('TREC', 'core17_%s.tsv' % topic)
        train, val, test = cls.splits(path, train=train_path, validation=dev_path, test=test_path)
        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)
        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)


class Robust05Hierarchical(Robust05):
    NESTING_FIELD = Field(batch_first=True, tokenize=clean_string)
    TEXT_FIELD = NestedField(NESTING_FIELD, tokenize=split_sents)