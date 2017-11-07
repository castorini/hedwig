from mp_cnn.evaluators.qa_evaluator import QAEvaluator


class WikiQAEvaluator(QAEvaluator):

    def __init__(self, dataset_cls, model, data_loader, batch_size, device):
        super(WikiQAEvaluator, self).__init__(dataset_cls, model, data_loader, batch_size, device)
