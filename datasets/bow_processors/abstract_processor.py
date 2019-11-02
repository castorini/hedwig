import csv
import sys

import torch
from torch import tensor
from torch.utils.data import Dataset


class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class BagOfWordsProcessor(object):
    def get_train_examples(self, data_dir):
        """
        Gets a collection of InputExamples for the train set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """
        Gets a collection of InputExamples for the dev set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """
        Gets a collection of InputExamples for the test set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """
        Reads a Tab Separated Values (TSV) file
        :param input_file:
        :param quotechar:
        :return:
        """
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(str(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class StreamingSparseDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        feature_tensor = tensor(self.features.getrow(idx).toarray(), dtype=torch.float)
        label_tensor = tensor(self.labels[idx], dtype=torch.long)
        return feature_tensor, label_tensor

    def __len__(self):
        return len(self.labels)
