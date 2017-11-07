from mp_cnn.trainers.sick_trainer import SICKTrainer
from mp_cnn.trainers.msrvid_trainer import MSRVIDTrainer
from mp_cnn.trainers.trecqa_trainer import TRECQATrainer
from mp_cnn.trainers.wikiqa_trainer import WikiQATrainer


class MPCNNTrainerFactory(object):
    """
    Get the corresponding Trainer class for a particular dataset.
    """
    trainer_map = {
        'sick': SICKTrainer,
        'msrvid': MSRVIDTrainer,
        'trecqa': TRECQATrainer,
        'wikiqa': WikiQATrainer
    }

    @staticmethod
    def get_trainer(dataset_name, model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):
        if dataset_name not in MPCNNTrainerFactory.trainer_map:
            raise ValueError('{} is not implemented.'.format(dataset_name))

        return MPCNNTrainerFactory.trainer_map[dataset_name](
            model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator
        )
