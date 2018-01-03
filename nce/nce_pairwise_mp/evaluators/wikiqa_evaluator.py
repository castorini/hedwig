from nce.nce_pairwise_mp.evaluators.qa_evaluator import QAEvaluator


class WikiQAEvaluatorNCE(QAEvaluator):

    def __init__(self, dataset_cls, model, data_loader, batch_size, device):
        super(WikiQAEvaluatorNCE, self).__init__(dataset_cls, model, data_loader, batch_size, device)
