from abc import ABCMeta, abstractmethod
import os

from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from torchtext.data.field import Field

from datasets.idf_utils import get_pairwise_word_to_doc_freq, get_pairwise_overlap_features


class CastorPairDataset(Dataset, metaclass=ABCMeta):

    # Child classes must define
    NAME = None
    NUM_CLASSES = None
    ID_FIELD = None
    TEXT_FIELD = None
    EXT_FEATS_FIELD = None
    LABEL_FIELD = None

    @abstractmethod
    def __init__(self, path):
        """
        Create a Castor dataset involving pairs of texts
        """
        fields = [('id', self.ID_FIELD), ('sentence_1', self.TEXT_FIELD), ('sentence_2', self.TEXT_FIELD), ('ext_feats', self.EXT_FEATS_FIELD), ('label', self.LABEL_FIELD)]

        examples = []
        with open(os.path.join(path, 'a.toks'), 'r') as f1, open(os.path.join(path, 'b.toks'), 'r') as f2:
            sent_list_1 = [l.rstrip('.\n').split(' ') for l in f1]
            sent_list_2 = [l.rstrip('.\n').split(' ') for l in f2]

        word_to_doc_cnt = get_pairwise_word_to_doc_freq(sent_list_1, sent_list_2)
        overlap_feats = get_pairwise_overlap_features(sent_list_1, sent_list_2, word_to_doc_cnt)

        with open(os.path.join(path, 'id.txt'), 'r') as id_file, open(os.path.join(path, 'sim.txt'), 'r') as label_file:
            for pair_id, l1, l2, ext_feats, label in zip(id_file, sent_list_1, sent_list_2, overlap_feats, label_file):
                pair_id = pair_id.rstrip('.\n')
                label = label.rstrip('.\n')
                example = Example.fromlist([pair_id, l1, l2, ext_feats, label], fields)
                examples.append(example)

        super(CastorPairDataset, self).__init__(examples, fields)
