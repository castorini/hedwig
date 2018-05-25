class Trainer(object):

    """
    Abstraction for training a model on a Dataset.
    """

    def __init__(self, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):
        self.model = model
        self.embedding = embedding
        self.optimizer = trainer_config['optimizer']
        self.train_loader = train_loader
        self.batch_size = trainer_config['batch_size']
        self.log_interval = trainer_config['log_interval']
        self.model_outfile = trainer_config['model_outfile']
        self.lr_reduce_factor = trainer_config['lr_reduce_factor']
        self.patience = trainer_config['patience']
        self.use_tensorboard = trainer_config['tensorboard']
        self.clip_norm = trainer_config.get('clip_norm')

        if self.use_tensorboard:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(log_dir=None, comment='' if trainer_config['run_label'] is None else trainer_config['run_label'])
        self.logger = trainer_config['logger']

        self.train_evaluator = train_evaluator
        self.test_evaluator = test_evaluator
        self.dev_evaluator = dev_evaluator

    def evaluate(self, evaluator, dataset_name):
        scores, metric_names = evaluator.get_scores()
        self.logger.info('Evaluation metrics for {}:'.format(dataset_name))
        self.logger.info('\t'.join([' '] + metric_names))
        self.logger.info('\t'.join([dataset_name] + list(map(str, scores))))
        return scores

    def get_sentence_embeddings(self, batch):
        sent1 = self.embedding(batch.sentence_1).transpose(1, 2)
        sent2 = self.embedding(batch.sentence_2).transpose(1, 2)
        return sent1, sent2

    def train_epoch(self, epoch):
        raise NotImplementedError()

    def train(self, epochs):
        raise NotImplementedError()
