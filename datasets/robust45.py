import csv
import os
import random
import re
import sys

import torch
from nltk import tokenize
from torchtext.data import NestedField, Field, TabularDataset
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors

csv.field_size_limit(sys.maxsize)


def clean_string(string, sentence_droprate=0, max_length=5000):
    """
    Performs tokenization and string cleaning
    """
    if sentence_droprate > 0:
        lines = [x for x in tokenize.sent_tokenize(string) if len(x) > 1]
        lines_drop = [x for x in lines if random.randint(0, 100) > 100 * sentence_droprate]
        string = ' '.join(lines_drop if len(lines_drop) > 0 else lines)

    string = re.sub(r'[^A-Za-z0-9]', ' ', string)
    string = re.sub(r'\s{2,}', ' ', string)
    tokenized_string = string.lower().strip().split()
    return tokenized_string[:min(max_length, len(tokenized_string))]


def split_sents(string, max_length=50):
    tokenized_string = [x for x in tokenize.sent_tokenize(string) if len(x) > 1]
    return tokenized_string[:min(max_length, len(tokenized_string))]


def process_labels(string):
    """
    Returns the label string as a list of integers
    :param string:
    :return:
    """
    return 0 if string == '01' else 1


def process_docids(string):
    """
    Returns the docid as an integer
    :param string:
    :return:
    """
    try:
        docid = int(string)
    except ValueError:
        # print("Error converting docid to integer:", string)
        docid = 0
    return docid


class Robust45(TabularDataset):
    NAME = 'Robust45'
    NUM_CLASSES = 2
    TEXT_FIELD = Field(batch_first=True, tokenize=clean_string, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_labels)
    DOCID_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_docids)
    TOPICS = ['307', '310', '321', '325', '330', '336', '341', '344', '345', '347', '350', '353', '354', '355', '356',
              '362', '363', '367', '372', '375', '378', '379', '389', '393', '394', '397', '399', '400', '404', '408',
              '414', '416', '419', '422', '423', '426', '427', '433', '435', '436', '439', '442', '443', '445', '614',
              '620', '626', '646', '677', '690']

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, path, train, validation, test, **kwargs):
        return super(Robust45, cls).splits(
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

        train_path = os.path.join('TREC', 'robust45_aug_train_%s.tsv' % topic)
        dev_path = os.path.join('TREC', 'robust45_dev_%s.tsv' % topic)
        test_path = os.path.join('TREC', 'core17_10k_%s.tsv' % topic)
        train, val, test = cls.splits(path, train=train_path, validation=dev_path, test=test_path)
        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)
        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)


class Robust45Hierarchical(Robust45):
    @staticmethod
    def clean_sentence(string):
        return clean_string(string, sentence_droprate=0, max_length=100)

    NESTING_FIELD = Field(batch_first=True, tokenize=clean_string)
    TEXT_FIELD = NestedField(NESTING_FIELD, tokenize=split_sents)