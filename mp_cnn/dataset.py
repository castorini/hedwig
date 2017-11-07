import os

import torch
import torch.nn as nn

from datasets.sick import SICK
from datasets.msrvid import MSRVID
from datasets.trecqa import TRECQA
from datasets.wikiqa import WikiQA


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


class MPCNNDatasetFactory(object):
    """
    Get the corresponding Dataset class for a particular dataset.
    """
    @staticmethod
    def get_dataset(dataset_name, word_vectors_dir, word_vectors_file, batch_size, device):
        if dataset_name == 'sick':
            dataset_root = os.path.join(os.pardir, os.pardir, 'data', 'sick/')
            train_loader, dev_loader, test_loader = SICK.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding_dim = SICK.TEXT_FIELD.vocab.vectors.size()
            embedding = nn.Embedding(embedding_dim[0], embedding_dim[1])
            embedding.weight = nn.Parameter(SICK.TEXT_FIELD.vocab.vectors)
            return SICK, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'msrvid':
            dataset_root = os.path.join(os.pardir, os.pardir, 'data', 'msrvid/')
            dev_loader = None
            train_loader, test_loader = MSRVID.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding_dim = MSRVID.TEXT_FIELD.vocab.vectors.size()
            embedding = nn.Embedding(embedding_dim[0], embedding_dim[1])
            embedding.weight = nn.Parameter(MSRVID.TEXT_FIELD.vocab.vectors)
            return MSRVID, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'trecqa':
            if not os.path.exists('../utils/trec_eval-9.0.5/trec_eval'):
                raise FileNotFoundError('TrecQA requires the trec_eval tool to run. Please run get_trec_eval.sh inside Castor/utils (as working directory) before continuing.')
            dataset_root = os.path.join(os.pardir, os.pardir, 'data', 'TrecQA/')
            train_loader, dev_loader, test_loader = TRECQA.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding_dim = TRECQA.TEXT_FIELD.vocab.vectors.size()
            embedding = nn.Embedding(embedding_dim[0], embedding_dim[1])
            embedding.weight = nn.Parameter(TRECQA.TEXT_FIELD.vocab.vectors)
            return TRECQA, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'wikiqa':
            if not os.path.exists('../utils/trec_eval-9.0.5/trec_eval'):
                raise FileNotFoundError('TrecQA requires the trec_eval tool to run. Please run get_trec_eval.sh inside Castor/utils (as working directory) before continuing.')
            dataset_root = os.path.join(os.pardir, os.pardir, 'data', 'WikiQA/')
            train_loader, dev_loader, test_loader = WikiQA.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding_dim = WikiQA.TEXT_FIELD.vocab.vectors.size()
            embedding = nn.Embedding(embedding_dim[0], embedding_dim[1])
            embedding.weight = nn.Parameter(WikiQA.TEXT_FIELD.vocab.vectors)
            return WikiQA, embedding, train_loader, test_loader, dev_loader
        else:
            raise ValueError('{} is not a valid dataset.'.format(dataset_name))

