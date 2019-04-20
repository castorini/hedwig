from common.evaluators.classification_evaluator import ClassificationEvaluator
from common.evaluators.relevance_transfer_evaluator import RelevanceTransferEvaluator


class EvaluatorFactory(object):
    """
    Get the corresponding Evaluator class for a particular dataset.
    """
    evaluator_map = {
        'Reuters': ClassificationEvaluator,
        'AAPD': ClassificationEvaluator,
        'IMDB': ClassificationEvaluator,
        'Yelp2014': ClassificationEvaluator,
        'Robust04': RelevanceTransferEvaluator,
        'Robust05': RelevanceTransferEvaluator,
        'Robust45': RelevanceTransferEvaluator
    }

    @staticmethod
    def get_evaluator(dataset_cls, model, embedding, data_loader, batch_size, device, keep_results=False):
        if data_loader is None:
            return None

        if not hasattr(dataset_cls, 'NAME'):
            raise ValueError('Invalid dataset. Dataset should have NAME attribute.')

        if dataset_cls.NAME not in EvaluatorFactory.evaluator_map:
            raise ValueError('{} is not implemented.'.format(dataset_cls))

        return EvaluatorFactory.evaluator_map[dataset_cls.NAME](
            dataset_cls, model, embedding, data_loader, batch_size, device, keep_results
        )
