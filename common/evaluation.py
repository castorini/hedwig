from .evaluators.sst_evaluator import SSTEvaluator
from .evaluators.reuters_evaluator import ReutersEvaluator


class EvaluatorFactory(object):
    """
    Get the corresponding Evaluator class for a particular dataset.
    """
    evaluator_map = {
        'SST-1': SSTEvaluator,
        'SST-2': SSTEvaluator,
        'Reuters': ReutersEvaluator,
        'AAPD': ReutersEvaluator,
        'IMDB': ReutersEvaluator,
        'Yelp2014': ReutersEvaluator
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
