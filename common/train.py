from .trainers.sick_trainer import SICKTrainer
from .trainers.msrvid_trainer import MSRVIDTrainer
from .trainers.trecqa_trainer import TRECQATrainer
from .trainers.wikiqa_trainer import WikiQATrainer
from nce.nce_pairwise_mp.trainers.trecqa_trainer import TRECQATrainerNCE
from nce.nce_pairwise_mp.trainers.wikiqa_trainer import WikiQATrainerNCE


class TrainerFactory(object):
    """
    Get the corresponding Trainer class for a particular dataset.
    """
    trainer_map = {
        'sick': SICKTrainer,
        'msrvid': MSRVIDTrainer,
        'trecqa': TRECQATrainer,
        'wikiqa': WikiQATrainer
    }

    trainer_map_nce = {
        'trecqa': TRECQATrainerNCE,
        'wikiqa': WikiQATrainerNCE
    }

    @staticmethod
    def get_trainer(dataset_name, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None, nce=False):
        if nce:
            trainer_map = TrainerFactory.trainer_map_nce
        else:
            trainer_map = TrainerFactory.trainer_map

        if dataset_name not in trainer_map:
            raise ValueError('{} is not implemented.'.format(dataset_name))

        return trainer_map[dataset_name](
            model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator
        )
