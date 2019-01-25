from .trainers.sick_trainer import SICKTrainer
from .trainers.msrvid_trainer import MSRVIDTrainer
from .trainers.trecqa_trainer import TRECQATrainer
from .trainers.wikiqa_trainer import WikiQATrainer
from .trainers.pit2015_trainer import PIT2015Trainer
from .trainers.sst_trainer import SSTTrainer
from .trainers.reuters_trainer import ReutersTrainer
from .trainers.snli_trainer import SNLITrainer
from .trainers.sts2014_trainer import STS2014Trainer
from .trainers.quora_trainer import QuoraTrainer
from nce.nce_pairwise_mp.trainers.trecqa_trainer import TRECQATrainerNCE
from nce.nce_pairwise_mp.trainers.wikiqa_trainer import WikiQATrainerNCE


class TrainerFactory(object):
    """
    Get the corresponding Trainer class for a particular dataset.
    """
    trainer_map = {
        'sick': SICKTrainer,
        'msrvid': MSRVIDTrainer,
        'SST-1': SSTTrainer,
        'SST-2': SSTTrainer,
        'trecqa': TRECQATrainer,
        'wikiqa': WikiQATrainer,
        'pit2015': PIT2015Trainer,
        'twitterurl': PIT2015Trainer,
        'Reuters': ReutersTrainer,
        'AAPD': ReutersTrainer,
        'IMDB': ReutersTrainer,
        'Yelp2014': ReutersTrainer,
        'snli': SNLITrainer,
        'sts2014': STS2014Trainer,
        'quora': QuoraTrainer
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
