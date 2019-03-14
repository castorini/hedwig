from .evaluators.sick_evaluator import SICKEvaluator
from .evaluators.msrvid_evaluator import MSRVIDEvaluator
from .evaluators.sst_evaluator import SSTEvaluator
from .evaluators.trecqa_evaluator import TRECQAEvaluator
from .evaluators.wikiqa_evaluator import WikiQAEvaluator
from .evaluators.pit2015_evaluator import PIT2015Evaluator
from .evaluators.reuters_evaluator import ReutersEvaluator
from .evaluators.snli_evaluator import SNLIEvaluator
from .evaluators.sts2014_evaluator import STS2014Evaluator
from .evaluators.quora_evaluator import QuoraEvaluator


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
    def get_evaluator(dataset_cls, model, embedding, data_loader, batch_size, device, nce=False, keep_results=False):
        if data_loader is None:
            return None

        if not hasattr(dataset_cls, 'NAME'):
            raise ValueError('Invalid dataset. Dataset should have NAME attribute.')

        if dataset_cls.NAME not in EvaluatorFactory.evaluator_map:
            raise ValueError('{} is not implemented.'.format(dataset_cls))

        return EvaluatorFactory.evaluator_map[dataset_cls.NAME](
            dataset_cls, model, embedding, data_loader, batch_size, device, keep_results
        )
