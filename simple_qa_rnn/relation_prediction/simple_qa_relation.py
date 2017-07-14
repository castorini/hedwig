import os

from torchtext import data

# most basic tokenizer - split on whitespace
def my_tokenizer():
    return lambda text: [tok for tok in text.split()]

class SimpleQaRelationDataset(data.ZipDataset, data.TabularDataset):

    url = 'https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz'
    filename = 'SimpleQuestions_v2.tgz'
    dirname = 'SimpleQuestions_v2'

    @staticmethod
    def sort_key(ex):
        return len(ex.question)

    @classmethod
    def splits(cls, text_field, label_field, root='../data',
                train='train.txt', validation='valid.txt', test='test.txt'):
        """Create dataset objects for splits of the Simple QA dataset.
        This is the most flexible way to use the dataset.
        Arguments:
            text_field: The field that will be used for premise and hypothesis
                data.
            label_field: The field that will be used for label data.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in which the
                train/valid/test data files will be stored.
            train: The filename of the train data. Default: 'annotated_fb_data_train.txt'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'annotated_fb_data_valid.txt'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'annotated_fb_data_test.txt'.
        """
        print("root path for relation dataset: {}".format(root))
        path = cls.download_or_unzip(root)
        prefix_fname = 'annotated_fb_data_'
        return super(SimpleQaRelationDataset, cls).splits(
                    os.path.join(path, prefix_fname), train, validation, test,
                    format='TSV', fields=[('subject', None), ('relation', label_field), (object, None), ('question', text_field)]
                )

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.', wv_dir='.',
                                wv_type=None, wv_dim='300d', **kwargs):
        """Create iterator objects for splits of the Simple QA dataset.
        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.
        Arguments:
            batch_size: Batch size.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-2
                subdirectory the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field(tokenize=my_tokenizer())
        LABEL = data.Field(sequential=False)

        train, val, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, wv_dir=wv_dir, wv_type=wv_type, wv_dim=wv_dim)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device)