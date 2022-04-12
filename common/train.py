from common.trainers.classification_trainer import ClassificationTrainer
from common.trainers.relevance_transfer_trainer import RelevanceTransferTrainer


class TrainerFactory(object):
    """
    Get the corresponding Trainer class for a particular dataset.
    """
    trainer_map = {
        'Reuters': ClassificationTrainer,
        'AAPD': ClassificationTrainer,
        'IMDB': ClassificationTrainer,
        'AG_NEWS': ClassificationTrainer,
        'DBpedia': ClassificationTrainer,
        'IMDB_torchtext': ClassificationTrainer,
        'SogouNews': ClassificationTrainer,
        'YahooAnswers': ClassificationTrainer,
        'YelpReviewPolarity': ClassificationTrainer,
        'TwentyNews': ClassificationTrainer,
        'R8': ClassificationTrainer,
        'R52': ClassificationTrainer,
        'OHSUMED': ClassificationTrainer,
        'TREC6': ClassificationTrainer,
        'Yelp2014': ClassificationTrainer,
        'Robust04': RelevanceTransferTrainer,
        'Robust05': RelevanceTransferTrainer,
        'Robust45': RelevanceTransferTrainer,
    }

    @staticmethod
    def get_trainer(dataset_name, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):

        if dataset_name not in TrainerFactory.trainer_map:
            raise ValueError('{} is not implemented.'.format(dataset_name))

        return TrainerFactory.trainer_map[dataset_name](
            model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator
        )
