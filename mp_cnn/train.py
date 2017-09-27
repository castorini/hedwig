import math
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr, spearmanr

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class MPCNNTrainerFactory(object):
    """
    Get the corresponding Trainer class for a particular dataset.
    """
    @staticmethod
    def get_trainer(dataset_name, model, optimizer, train_loader, batch_size, sample, log_interval, model_outfile, lr_reduce_factor, patience, train_evaluator, test_evaluator, dev_evaluator=None):
        if dataset_name == 'sick':
            return SICKTrainer(model, optimizer, train_loader, batch_size, sample, log_interval, model_outfile, lr_reduce_factor, patience, train_evaluator, test_evaluator, dev_evaluator)
        elif dataset_name == 'msrvid':
            return MSRVIDTrainer(model, optimizer, train_loader, batch_size, sample, log_interval, model_outfile,lr_reduce_factor, patience, train_evaluator, test_evaluator, dev_evaluator)
        else:
            raise ValueError('{} is not a valid dataset.'.format(dataset_name))


class Trainer(object):

    """
    Abstraction for training a model on a Dataset.
    """

    def __init__(self, model, optimizer, train_loader, batch_size, sample, log_interval, model_outfile, lr_reduce_factor, patience, train_evaluator, test_evaluator, dev_evaluator=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.batch_size = batch_size
        self.sample = sample
        self.log_interval = log_interval
        self.model_outfile = model_outfile
        self.lr_reduce_factor = lr_reduce_factor
        self.patience = patience
        self.train_evaluator = train_evaluator
        self.test_evaluator = test_evaluator
        self.dev_evaluator = dev_evaluator

    def evaluate(self, evaluator, dataset_name):
        scores, metric_names = evaluator.get_scores()
        logger.info('Evaluation metrics for {}:'.format(dataset_name))
        logger.info('\t'.join([' '] + metric_names))
        logger.info('\t'.join([dataset_name] + list(map(str, scores))))
        return scores

    def train_epoch(self, epoch):
        raise NotImplementedError()

    def train(self, epochs):
        raise NotImplementedError()


class SICKTrainer(Trainer):

    def __init__(self, model, optimizer, train_loader, batch_size, sample, log_interval, model_outfile, lr_reduce_factor, patience, train_evaluator, test_evaluator, dev_evaluator=None):
        super(SICKTrainer, self).__init__(model, optimizer, train_loader, batch_size, sample, log_interval, model_outfile, lr_reduce_factor, patience, train_evaluator, test_evaluator, dev_evaluator)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (sentences, labels) in enumerate(self.train_loader):
            sent_a, sent_b = Variable(sentences['a']), Variable(sentences['b'])
            ext_feats = Variable(sentences['ext_feats'])
            labels = Variable(labels)
            self.optimizer.zero_grad()
            output = self.model(sent_a, sent_b, ext_feats)
            loss = F.kl_div(output, labels)
            total_loss += loss.data[0]
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, min(batch_idx * self.batch_size, len(self.train_loader.dataset)),
                    len(self.train_loader.dataset) if not self.sample else self.sample,
                    100. * batch_idx / (len(self.train_loader) if not self.sample else math.ceil(self.sample / self.batch_size)), loss.data[0])
                )

            del loss, output

        return total_loss

    def train(self, epochs):
        scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.lr_reduce_factor, patience=self.patience)
        epoch_times = []
        prev_loss = -1
        best_dev_score = -1
        for epoch in range(1, epochs + 1):
            start = time.time()
            logger.info('Epoch {} started...'.format(epoch))
            self.train_epoch(epoch)

            dev_scores = self.evaluate(self.dev_evaluator, 'dev')
            new_loss = dev_scores[2]
            end = time.time()
            duration = end - start
            logger.info('Epoch {} finished in {:.2f} minutes'.format(epoch, duration / 60))
            epoch_times.append(duration)

            if dev_scores[0] > best_dev_score:
                best_dev_score = dev_scores[0]
                torch.save(self.model, self.model_outfile)

            if abs(prev_loss - new_loss) <= 0.0002:
                logger.info('Early stopping. Loss changed by less than 0.0002.')
                break

            prev_loss = new_loss
            scheduler.step(dev_scores[0])

        logger.info('Training took {:.2f} minutes overall...'.format(sum(epoch_times) / 60))


class MSRVIDTrainer(Trainer):

    def __init__(self, model, optimizer, train_loader, batch_size, sample, log_interval, model_outfile, lr_reduce_factor, patience, train_evaluator, test_evaluator, dev_evaluator=None):
        super(MSRVIDTrainer, self).__init__(model, optimizer, train_loader, batch_size, sample, log_interval, model_outfile, lr_reduce_factor, patience, train_evaluator, test_evaluator, dev_evaluator)

    def train_epoch(self, epoch):
        self.model.train()

        # since MSRVID doesn't have validation set, we manually leave-out some training data for validation
        batches = math.ceil(len(self.train_loader.dataset) / self.batch_size)
        start_val_batch = math.floor(0.8 * batches)
        left_out_val_a, left_out_val_b = [], []
        left_out_ext_feats = []
        left_out_val_labels = []

        for batch_idx, (sentences, labels) in enumerate(self.train_loader):
            sent_a, sent_b = Variable(sentences['a']), Variable(sentences['b'])
            ext_feats = Variable(sentences['ext_feats'])
            labels = Variable(labels)
            if batch_idx >= start_val_batch:
                left_out_val_a.append(sent_a)
                left_out_val_b.append(sent_b)
                left_out_val_labels.append(labels)
                continue
            self.optimizer.zero_grad()
            output = self.model(sent_a, sent_b, ext_feats)
            loss = F.kl_div(output, labels)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, min(batch_idx * self.batch_size, len(self.train_loader.dataset)),
                    len(self.train_loader.dataset) if not self.sample else self.sample,
                    100. * batch_idx / (len(self.train_loader) if not self.sample else math.ceil(self.sample / self.batch_size)), loss.data[0])
                )

            del loss, output

        self.evaluate(self.train_evaluator, 'train')
        return left_out_val_a, left_out_val_b, left_out_ext_feats, left_out_val_labels

    def train(self, epochs):
        scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.lr_reduce_factor, patience=self.patience)
        epoch_times = []
        prev_loss = -1
        best_dev_score = -1
        for epoch in range(1, epochs + 1):
            start = time.time()
            logger.info('Epoch {} started...'.format(epoch))
            left_out_a, left_out_b, left_out_ext_feats, left_out_label = self.train_epoch(epoch)

            # manually evaluating the validating set
            left_out_a = torch.cat(left_out_a)
            left_out_b = torch.cat(left_out_b)
            left_out_ext_feats = torch.cat(left_out_ext_feats)
            left_out_label = torch.cat(left_out_label)
            output = self.model(left_out_a, left_out_b, left_out_ext_feats)
            val_kl_div_loss = F.kl_div(output, left_out_label).data[0]
            predict_classes = torch.arange(0, 6).expand(len(left_out_a), 6).cuda()
            true_labels = (predict_classes * left_out_label.data).sum(dim=1)
            predictions = (predict_classes * output.data.exp()).sum(dim=1)
            predictions = predictions.cpu().numpy()
            true_labels = true_labels.cpu().numpy()
            pearson_r = pearsonr(predictions, true_labels)[0]
            for param_group in self.optimizer.param_groups:
                logger.info('Validation size: %s Pearson\'s r: %s', output.size()[0], pearson_r)
                logger.info('Learning rate: %s', param_group['lr'])
                break
            scheduler.step(pearson_r)

            end = time.time()
            duration = end - start
            logger.info('Epoch {} finished in {:.2f} minutes'.format(epoch, duration / 60))
            epoch_times.append(duration)

            if pearson_r > best_dev_score:
                best_dev_score = pearson_r
                torch.save(self.model, self.model_outfile)

            if abs(prev_loss - val_kl_div_loss) <= 0.0005:
                logger.info('Early stopping. Loss changed by less than 0.0005.')
                break

            prev_loss = val_kl_div_loss
            self.evaluate(self.test_evaluator, 'test')

        logger.info('Training took {:.2f} minutes overall...'.format(sum(epoch_times) / 60))
