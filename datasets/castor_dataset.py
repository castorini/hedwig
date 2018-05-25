from abc import ABCMeta, abstractmethod
import os
from sys import exit

import numpy as np
import torch
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example

from datasets.idf_utils import get_pairwise_word_to_doc_freq, get_pairwise_overlap_features


class CastorPairDataset(Dataset, metaclass=ABCMeta):

    # Child classes must define
    NAME = None
    NUM_CLASSES = None
    ID_FIELD = None
    TEXT_FIELD = None
    EXT_FEATS_FIELD = None
    LABEL_FIELD = None
    RAW_TEXT_FIELD = None
    EXT_FEATS = 4
    AID_FIELD = None

    @abstractmethod
    def __init__(self, path, load_ext_feats=False):
        """
        Create a Castor dataset involving pairs of texts
        """
        fields = [('id', self.ID_FIELD), ('sentence_1', self.TEXT_FIELD), ('sentence_2', self.TEXT_FIELD),
                  ('ext_feats', self.EXT_FEATS_FIELD), ('label', self.LABEL_FIELD),
                  ('aid', self.AID_FIELD), ('sentence_1_raw', self.RAW_TEXT_FIELD), ('sentence_2_raw', self.RAW_TEXT_FIELD)]

        examples = []
        with open(os.path.join(path, 'a.toks'), 'r') as f1, open(os.path.join(path, 'b.toks'), 'r') as f2:
            sent_list_1 = [l.rstrip('.\n').split(' ') for l in f1]
            sent_list_2 = [l.rstrip('.\n').split(' ') for l in f2]

        word_to_doc_cnt = get_pairwise_word_to_doc_freq(sent_list_1, sent_list_2)
        self.word_to_doc_cnt = word_to_doc_cnt

        if not load_ext_feats:
            overlap_feats = get_pairwise_overlap_features(sent_list_1, sent_list_2, word_to_doc_cnt)
        else:
            overlap_feats = np.loadtxt(os.path.join(path, 'overlap_feats.txt'))

        with open(os.path.join(path, 'id.txt'), 'r') as id_file, open(os.path.join(path, 'sim.txt'), 'r') as label_file:
            for i, (pair_id, l1, l2, ext_feats, label) in enumerate(zip(id_file, sent_list_1, sent_list_2, overlap_feats, label_file)):
                pair_id = pair_id.rstrip('.\n')
                label = label.rstrip('.\n')
                example_list = [pair_id, l1, l2, ext_feats, label, i + 1, ' '.join(l1), ' '.join(l2)]
                example = Example.fromlist(example_list, fields)
                examples.append(example)

        super(CastorPairDataset, self).__init__(examples, fields)

    @classmethod
    def set_vectors(cls, field, vector_path):
        if os.path.isfile(vector_path):
            stoi, vectors, dim = torch.load(vector_path)
            field.vocab.vectors = torch.Tensor(len(field.vocab), dim)

            for i, token in enumerate(field.vocab.itos):
                wv_index = stoi.get(token, None)
                if wv_index is not None:
                    field.vocab.vectors[i] = vectors[wv_index]
                else:
                    # initialize <unk> with uniform_(-0.05, 0.05) vectors
                    field.vocab.vectors[i] = torch.FloatTensor(dim).uniform_(-0.05, 0.05)
        else:
            print("Error: Need word embedding pt file")
            exit(1)
        return field
