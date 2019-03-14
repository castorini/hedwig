from torchtext import data

class WikiDataset(data.TabularDataset):
    dirname = 'data'
    @classmethod

    def splits(cls, question_id, question_field, answer_field, external_field, label_field, root='.data',
               train='wikiqa.train.tsv', validation='wikiqa.dev.tsv', test='wikiqa.test.tsv'):
        path = './data'
        return super(WikiDataset, cls).splits(
            path, root, train, validation, test,
            format='TSV', fields=[('qid', question_id), ('label', label_field), ('question', question_field),
                                  ('answer', answer_field), ('ext_feat', external_field)]
        )
