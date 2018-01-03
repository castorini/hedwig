from nce.nce_pairwise_mp.trainers.qa_trainer import QATrainer


class TRECQATrainerNCE(QATrainer):

    def __init__(self, model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):
        super(TRECQATrainerNCE, self).__init__(model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)
