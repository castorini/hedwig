from mp_cnn.trainers.qa_trainer import QATrainer


class WikiQATrainer(QATrainer):

    def __init__(self, model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):
        super(WikiQATrainer, self).__init__(model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)
