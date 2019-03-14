import os

import torch
import torch.nn as nn

from datasets.sick import SICK
from datasets.msrvid import MSRVID
from datasets.trecqa import TRECQA
from datasets.wikiqa import WikiQA
from datasets.pit2015 import PIT2015
from datasets.snli import SNLI
from datasets.sts2014 import STS2014
from datasets.quora import Quora
from datasets.reuters import Reuters
from datasets.aapd import AAPD
from datasets.imdb import IMDB


class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].normal_(0, 0.01)
        return cls.cache[size_tup]


class DatasetFactory(object):
    """
    Get the corresponding Dataset class for a particular dataset.
    """
    @staticmethod
    def get_dataset(dataset_name, word_vectors_dir, word_vectors_file, batch_size, device, castor_dir="./", utils_trecqa="utils/trec_eval-9.0.5/trec_eval"):
        if dataset_name == 'sick':
            dataset_root = os.path.join(castor_dir, os.pardir, 'Castor-data', 'datasets', 'sick/')
            train_loader, dev_loader, test_loader = SICK.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(SICK.TEXT_FIELD.vocab.vectors)
            return SICK, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'msrvid':
            dataset_root = os.path.join(castor_dir, os.pardir, 'Castor-data', 'datasets', 'msrvid/')
            dev_loader = None
            train_loader, test_loader = MSRVID.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(MSRVID.TEXT_FIELD.vocab.vectors)
            return MSRVID, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'trecqa':
            if not os.path.exists(os.path.join(castor_dir, utils_trecqa)):
                raise FileNotFoundError('TrecQA requires the trec_eval tool to run. Please run get_trec_eval.sh inside Castor/utils (as working directory) before continuing.')
            dataset_root = os.path.join(castor_dir, os.pardir, 'Castor-data', 'datasets', 'TrecQA/')
            train_loader, dev_loader, test_loader = TRECQA.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(TRECQA.TEXT_FIELD.vocab.vectors)
            return TRECQA, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'wikiqa':
            if not os.path.exists(os.path.join(castor_dir, utils_trecqa)):
                raise FileNotFoundError('WikiQA requires the trec_eval tool to run. Please run get_trec_eval.sh inside Castor/utils (as working directory) before continuing.')
            dataset_root = os.path.join(castor_dir, os.pardir, 'Castor-data', 'datasets', 'WikiQA/')
            train_loader, dev_loader, test_loader = WikiQA.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(WikiQA.TEXT_FIELD.vocab.vectors)
            return WikiQA, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'pit2015':
            dataset_root = os.path.join(castor_dir, os.pardir, 'Castor-data', 'datasets', 'SemEval-PIT2015/')
            train_loader, dev_loader, test_loader = PIT2015.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(PIT2015.TEXT_FIELD.vocab.vectors)
            return PIT2015, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'twitterurl':
            dataset_root = os.path.join(castor_dir, os.pardir, 'Castor-data', 'datasets', 'Twitter-URL/')
            train_loader, dev_loader, test_loader = PIT2015.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(PIT2015.TEXT_FIELD.vocab.vectors)
            return PIT2015, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'snli':
            dataset_root = os.path.join(castor_dir, os.pardir, 'Castor-data', 'datasets', 'snli_1.0/')
            train_loader, dev_loader, test_loader = SNLI.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(SNLI.TEXT_FIELD.vocab.vectors)
            return SNLI, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'sts2014':
            dataset_root = os.path.join(castor_dir, os.pardir, 'Castor-data', 'datasets', 'STS-2014')
            train_loader, dev_loader, test_loader = STS2014.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(STS2014.TEXT_FIELD.vocab.vectors)
            return STS2014, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == "quora":
            dataset_root = os.path.join(castor_dir, os.pardir, 'Castor-data', 'datasets', 'quora/')
            train_loader, dev_loader, test_loader = Quora.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(Quora.TEXT_FIELD.vocab.vectors)
            return Quora, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'reuters':
            dataset_root = os.path.join(castor_dir, os.pardir, 'Castor-data', 'datasets', 'Reuters-21578/')
            train_loader, dev_loader, test_loader = Reuters.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(Reuters.TEXT_FIELD.vocab.vectors)
            return Reuters, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'aapd':
            dataset_root = os.path.join(castor_dir, os.pardir, 'Castor-data', 'datasets', 'AAPD/')
            train_loader, dev_loader, test_loader = AAPD.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(AAPD.TEXT_FIELD.vocab.vectors)
            return AAPD, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'imdb':
            dataset_root = os.path.join(castor_dir, os.pardir, 'Castor-data', 'datasets', 'IMDB/')
            train_loader, dev_loader, test_loader = AAPD.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(AAPD.TEXT_FIELD.vocab.vectors)
            return IMDB, embedding, train_loader, test_loader, dev_loader
        else:
            raise ValueError('{} is not a valid dataset.'.format(dataset_name))

