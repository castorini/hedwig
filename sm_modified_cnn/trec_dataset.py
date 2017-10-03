from torchtext import data
import os

class TrecDataset(data.TabularDataset):
    dirname = 'data'
    @classmethod

    def splits(cls, question_id, question_field, answer_field, external_field, label_field,
               train='train.tsv', validation='dev.tsv', test='test.tsv'):
        path = './data'
        prefix_name = 'trecqa.'
        return super(TrecDataset, cls).splits(
            os.path.join(path, prefix_name), train, validation, test,
            format='TSV', fields=[('qid', question_id), ('label', label_field), ('question', question_field),
                                  ('answer', answer_field), ('ext_feat', external_field)]
        )
