from nce.nce_pairwise_mp.trainers.qa_trainer import QATrainer


class WikiQATrainerNCE(QATrainer):

    def __init__(self, model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):
        super(WikiQATrainerNCE, self).__init__(model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)
