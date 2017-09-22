from collections import defaultdict
from enum import Enum
import math
import os

import nltk
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data

import preprocessing

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class DatasetType(Enum):
    TRAIN = 1
    TEST = 2
    DEV = 3


class MPCNNDatasetFactory(object):
    """
    Get the corresponding Dataset class for a particular dataset.
    """
    @staticmethod
    def get_dataset(dataset_name, word_vectors_file, batch_size, cuda, sample):
        extra_args = {'shuffle': True}
        dev_loader = None
        if sample:
            sample_indices = list(range(sample))
            subset_random_sampler = data.sampler.SubsetRandomSampler(sample_indices)
            extra_args['sampler'] = subset_random_sampler
            extra_args['shuffle'] = False
        if dataset_name == 'sick':
            train_loader = torch.utils.data.DataLoader(SICKDataset(DatasetType.TRAIN, cuda), batch_size=batch_size, **extra_args)
            test_loader = torch.utils.data.DataLoader(SICKDataset(DatasetType.TEST, cuda), batch_size=batch_size, **extra_args)
            dev_loader = torch.utils.data.DataLoader(SICKDataset(DatasetType.DEV, cuda), batch_size=batch_size, **extra_args)
        elif dataset_name == 'msrvid':
            train_loader = torch.utils.data.DataLoader(MSRVIDDataset(DatasetType.TRAIN, cuda), batch_size=batch_size, **extra_args)
            test_loader = torch.utils.data.DataLoader(MSRVIDDataset(DatasetType.TEST, cuda), batch_size=batch_size, **extra_args)
        else:
            raise ValueError('{} is not a valid dataset.'.format(dataset_name))

        word_index, embedding = preprocessing.get_glove_embedding(word_vectors_file, train_loader.dataset.dataset_root)
        logger.info('Finished loading GloVe embedding for vocab in data...')

        train_loader.dataset.initialize(word_index, embedding)
        test_loader.dataset.initialize(word_index, embedding)
        if dev_loader is not None:
            dev_loader.dataset.initialize(word_index, embedding)
        return train_loader, test_loader, dev_loader


class MPCNNDataset(data.Dataset):
    train_folder = 'train'
    test_folder = 'test'
    dev_folder = 'dev'
    # subclass will override fields below
    dataset_root = None
    num_classes = None

    def __init__(self, dataset_type, cuda):
        if not isinstance(dataset_type, DatasetType):
            raise ValueError('dataset_type ({}) must be of type DatasetType enum'.format(dataset_type))

        if dataset_type == DatasetType.TRAIN:
            subfolder = MPCNNDataset.train_folder
        elif dataset_type == DatasetType.TEST:
            subfolder = MPCNNDataset.test_folder
        else:
            subfolder = MPCNNDataset.dev_folder

        self.dataset_dir = os.path.join(self.dataset_root, subfolder)
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError('{} does not exist'.format(self.dataset_dir))

        self.cuda = cuda
        self.max_length = -10000
        self.unk = torch.Tensor(300)
        self.unk.normal_(0, 0.01)

    def initialize(self, word_index, embedding):
        """
        Convert sentences into sentence embeddings.
        """
        sent_a = self._load(self.dataset_dir, 'a.txt')
        sent_b = self._load(self.dataset_dir, 'b.txt')
        word_to_doc_cnt = defaultdict(int)

        # obtain max sentence length to use as dimension for padding to support batching
        sent_a_tokens, sent_b_tokens = [], []
        for i in range(len(sent_a)):
            sa_tokens = sent_a[i].split(' ')
            sb_tokens = sent_b[i].split(' ')
            self.max_length = max(self.max_length, len(sa_tokens), len(sb_tokens))
            sent_a_tokens.append(sa_tokens)
            sent_b_tokens.append(sb_tokens)

            unique_tokens = set(sa_tokens) | set(sb_tokens)
            for t in unique_tokens:
                word_to_doc_cnt[t] += 1

        self.sentences = []
        stoplist = set(stopwords.words('english'))
        num_docs = len(word_to_doc_cnt)
        for i in range(len(sent_a)):
            sent_pair = {}
            sent_pair['a'] = self._get_sentence_embeddings(sent_a_tokens[i], word_index, embedding)
            sent_pair['b'] = self._get_sentence_embeddings(sent_b_tokens[i], word_index, embedding)

            tokens_a_set, tokens_b_set = set(sent_a_tokens[i]), set(sent_b_tokens[i])
            intersect = tokens_a_set & tokens_b_set
            overlap = len(intersect) / (len(tokens_a_set) + len(tokens_b_set))
            idf_intersect = sum(np.math.log(num_docs / word_to_doc_cnt[w]) for w in intersect)
            idf_weighted_overlap = idf_intersect / (len(tokens_a_set) + len(tokens_b_set))

            tokens_a_set_no_stop = set(w for w in sent_a_tokens[i] if w not in stoplist)
            tokens_b_set_no_stop = set(w for w in sent_b_tokens[i] if w not in stoplist)
            intersect_no_stop = tokens_a_set_no_stop & tokens_b_set_no_stop
            overlap_no_stop = len(intersect_no_stop) / (len(tokens_a_set_no_stop) + len(tokens_b_set_no_stop))
            idf_intersect_no_stop = sum(np.math.log(num_docs / word_to_doc_cnt[w]) for w in intersect_no_stop)
            idf_weighted_overlap_no_stop = idf_intersect_no_stop / (len(tokens_a_set_no_stop) + len(tokens_b_set_no_stop))
            ext_feats = torch.Tensor([overlap, idf_weighted_overlap, overlap_no_stop, idf_weighted_overlap_no_stop])
            ext_feats = ext_feats.cuda() if self.cuda else ext_feats
            sent_pair['ext_feats'] = ext_feats

            self.sentences.append(sent_pair)

        self.labels = self._load(self.dataset_dir, 'sim.txt', float)

    def _load(self, dataset_dir, fname, type_converter=str):
        data = []
        with open(os.path.join(dataset_dir, fname), 'r') as f:
            for line in f:
                stripped_line = line.rstrip('.\n')
                item = type_converter(stripped_line)
                data.append(item)
        return data

    def _get_sentence_embeddings(self, tokens, word_index, embedding):
        sentence_embedding = torch.zeros(300, self.max_length)
        sentence_embedding[:, :len(tokens)].normal_(0, 1)
        found_pos, found_emb_idx = [], []
        for i, token in enumerate(tokens):
            if token in word_index:
                found_pos.append(i)
                found_emb_idx.append(word_index[token])
            else:
                sentence_embedding[:, i] = self.unk

        found_word_vecs = embedding(Variable(torch.LongTensor(found_emb_idx)))
        for i, v in enumerate(found_pos):
            sentence_embedding[:, v] = found_word_vecs[i].data
        return sentence_embedding.cuda() if self.cuda else sentence_embedding

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class SICKDataset(MPCNNDataset):

    dataset_root = os.path.join(os.pardir, os.pardir, 'data', 'sick')
    num_classes = 5

    def __init__(self, dataset_type, cuda):
        super(SICKDataset, self).__init__(dataset_type, cuda)

    def initialize(self, word_index, embedding):
        super(SICKDataset, self).initialize(word_index, embedding)
        new_labels = torch.zeros(self.__len__(), self.num_classes)
        for i, sim in enumerate(self.labels):
            ceil, floor = math.ceil(sim), math.floor(sim)
            if ceil == floor:
                new_labels[i][floor - 1] = 1
            else:
                new_labels[i][floor - 1] = ceil - sim
                new_labels[i][ceil - 1] = sim - floor

        self.labels = new_labels.cuda() if self.cuda else new_labels


class MSRVIDDataset(MPCNNDataset):

    dataset_root = os.path.join(os.pardir, os.pardir, 'data', 'msrvid')
    num_classes = 6

    def __init__(self, dataset_type, cuda):
        super(MSRVIDDataset, self).__init__(dataset_type, cuda)

    def initialize(self, word_index, embedding):
        super(MSRVIDDataset, self).initialize(word_index, embedding)
        new_labels = torch.zeros(self.__len__(), self.num_classes)
        for i, sim in enumerate(self.labels):
            ceil, floor = math.ceil(sim), math.floor(sim)
            if ceil == floor:
                new_labels[i][floor] = 1
            else:
                new_labels[i][floor] = ceil - sim
                new_labels[i][ceil] = sim - floor

        self.labels = new_labels.cuda() if self.cuda else new_labels
