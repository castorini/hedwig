from torchtext import data

class TrecDataset(data.TabularDataset):
    dirname = 'data'
    @classmethod

    def splits(cls, question_id, question_field, answer_field, external_field, label_field, root='.data',
               train='trecqa.train.tsv', validation='trecqa.dev.tsv', test='trecqa.test.tsv'):
        path = './data'
        return super(TrecDataset, cls).splits(
            path, root, train, validation, test,
            format='TSV', fields=[('qid', question_id), ('label', label_field), ('question', question_field),
                                  ('answer', answer_field), ('ext_feat', external_field)]
        )
