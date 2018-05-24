from .qa_trainer import QATrainer


class TRECQATrainer(QATrainer):

    def __init__(self, model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):
        super(TRECQATrainer, self).__init__(model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)
