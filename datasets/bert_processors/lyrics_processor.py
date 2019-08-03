import os

from datasets.bert_processors.abstract_processor import BertProcessor, InputExample


class LyricsProcessor(BertProcessor):
    def __init__(self):
        self.NAME = 'Lyrics'

    def set_num_classes_(self, data_dir):
        with open(os.path.join(data_dir, 'Lyrics', 'train.tsv'), 'r') as f:
            l1 = f.readline().split('\t')

        # from one-hot class vector
        self.NUM_CLASSES = len(l1[0])
        self.IS_MULTILABEL = True #self.NUM_CLASSES > 2

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'Lyrics', 'train.tsv')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'Lyrics', 'dev.tsv')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'Lyrics', 'test.tsv')), 'test')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
