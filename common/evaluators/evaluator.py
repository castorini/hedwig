class Evaluator(object):
    """
    Evaluates performance of model on a Dataset, using metrics specific to the Dataset.
    """

    def __init__(self, dataset_cls, model, data_loader, batch_size, device):
        self.dataset_cls = dataset_cls
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.device = device

    def get_scores(self):
        """
        Get the scores used to evaluate the model.
        Should return ([score1, score2, ..], [score1_name, score2_name, ...]).
        The first score is the primary score used to determine if the model has improved.
        """
        raise NotImplementedError('Evaluator subclass needs to implement get_score')
