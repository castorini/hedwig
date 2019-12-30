import os

from datasets.bert_processors.abstract_processor import BertProcessor, InputExample, InputFeatures


class RelevanceFeatures(InputFeatures):
    """A single set of features for relevance tasks."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, guid):
        super().__init__(input_ids, input_mask, segment_ids, label_id)
        self.guid = guid


class Robust45Processor(BertProcessor):
    NAME = 'Robust45'
    NUM_CLASSES = 2
    TOPICS = ['307', '310', '321', '325', '330', '336', '341', '344', '345', '347', '350', '353', '354', '355', '356',
              '362', '363', '367', '372', '375', '378', '379', '389', '393', '394', '397', '399', '400', '404', '408',
              '414', '416', '419', '422', '423', '426', '427', '433', '435', '436', '439', '442', '443', '445', '614',
              '620', '626', '646', '677', '690']

    def get_train_examples(self, data_dir, **kwargs):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'TREC', 'robust45_aug_train_%s.tsv' % kwargs['topic'])), 'train')

    def get_dev_examples(self, data_dir, **kwargs):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'TREC', 'robust45_dev_%s.tsv' % kwargs['topic'])), 'dev')

    def get_test_examples(self, data_dir, **kwargs):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'TREC', 'core17_10k_%s.tsv' % kwargs['topic'])), 'test')

    @staticmethod
    def _create_examples(lines, split):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[2]
            guid = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :return: a list of InputBatch objects
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        try:
            docid = int(example.guid)
        except ValueError:
            # print("Error converting docid to integer:", string)
            docid = 0

        features.append(RelevanceFeatures(input_ids=input_ids,
                                          input_mask=input_mask,
                                          segment_ids=segment_ids,
                                          label_id=0 if example.label == '01' else 1,
                                          guid=docid))
    return features
