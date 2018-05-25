from .evaluators.sick_evaluator import SICKEvaluator
from .evaluators.msrvid_evaluator import MSRVIDEvaluator
from .evaluators.trecqa_evaluator import TRECQAEvaluator
from .evaluators.wikiqa_evaluator import WikiQAEvaluator
from nce.nce_pairwise_mp.evaluators.trecqa_evaluator import TRECQAEvaluatorNCE
from nce.nce_pairwise_mp.evaluators.wikiqa_evaluator import WikiQAEvaluatorNCE

class EvaluatorFactory(object):
    """
    Get the corresponding Evaluator class for a particular dataset.
    """
    evaluator_map = {
        'sick': SICKEvaluator,
        'msrvid': MSRVIDEvaluator,
        'trecqa': TRECQAEvaluator,
        'wikiqa': WikiQAEvaluator
    }

    evaluator_map_nce = {
        'trecqa': TRECQAEvaluatorNCE,
        'wikiqa': WikiQAEvaluatorNCE
    }

    @staticmethod
    def get_evaluator(dataset_cls, model, embedding, data_loader, batch_size, device, nce=False):
        if data_loader is None:
            return None

        if nce:
            evaluator_map = EvaluatorFactory.evaluator_map_nce
        else:
            evaluator_map = EvaluatorFactory.evaluator_map

        if not hasattr(dataset_cls, 'NAME'):
            raise ValueError('Invalid dataset. Dataset should have NAME attribute.')

        if dataset_cls.NAME not in evaluator_map:
            raise ValueError('{} is not implemented.'.format(dataset_cls))

        return evaluator_map[dataset_cls.NAME](
            dataset_cls, model, embedding, data_loader, batch_size, device
        )
