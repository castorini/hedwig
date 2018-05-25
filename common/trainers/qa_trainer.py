import time

import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .trainer import Trainer
from utils.serialization import save_checkpoint


class QATrainer(Trainer):

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # Select embedding
            sent1, sent2 = self.get_sentence_embeddings(batch)

            output = self.model(sent1, sent2, batch.ext_feats, batch.dataset.word_to_doc_cnt, batch.sentence_1_raw, batch.sentence_2_raw)
            loss = F.nll_loss(output, batch.label, size_average=False)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, min(batch_idx * self.batch_size, len(batch.dataset.examples)),
                    len(batch.dataset.examples),
                    100. * batch_idx / (len(self.train_loader)), loss.item() / len(batch))
                )

        mean_average_precision, mean_reciprocal_rank, average_loss = self.evaluate(self.train_evaluator, 'train')

        if self.use_tensorboard:
            self.writer.add_scalar('{}/train/cross_entropy_loss'.format(self.train_loader.dataset.NAME), average_loss, epoch)
            self.writer.add_scalar('{}/train/map'.format(self.train_loader.dataset.NAME), mean_average_precision, epoch)
            self.writer.add_scalar('{}/train/mrr'.format(self.train_loader.dataset.NAME), mean_reciprocal_rank, epoch)

        return total_loss

    def train(self, epochs):
        scheduler = None
        if self.lr_reduce_factor != 1 and self.lr_reduce_factor != None:
            scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.lr_reduce_factor, patience=self.patience)
        epoch_times = []
        prev_loss = -1
        best_dev_score = -1
        for epoch in range(1, epochs + 1):
            start = time.time()
            self.logger.info('Epoch {} started...'.format(epoch))
            self.train_epoch(epoch)

            dev_scores = self.evaluate(self.dev_evaluator, 'dev')
            mean_average_precision, mean_reciprocal_rank, new_loss = dev_scores

            if self.use_tensorboard:
                self.writer.add_scalar('{}/lr'.format(self.train_loader.dataset.NAME), self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('{}/dev/cross_entropy_loss'.format(self.train_loader.dataset.NAME), new_loss, epoch)
                self.writer.add_scalar('{}/dev/map'.format(self.train_loader.dataset.NAME), mean_average_precision, epoch)
                self.writer.add_scalar('{}/dev/mrr'.format(self.train_loader.dataset.NAME), mean_reciprocal_rank, epoch)

            end = time.time()
            duration = end - start
            self.logger.info('Epoch {} finished in {:.2f} minutes'.format(epoch, duration / 60))
            epoch_times.append(duration)

            if mean_average_precision > best_dev_score:
                best_dev_score = mean_average_precision
                save_checkpoint(epoch, self.model.arch, self.model.state_dict(), self.optimizer.state_dict(), best_dev_score, self.model_outfile)

            if abs(prev_loss - new_loss) <= 0.0002:
                self.logger.info('Early stopping. Loss changed by less than 0.0002.')
                break

            prev_loss = new_loss
            if scheduler is not None:
                scheduler.step(mean_average_precision)

        self.logger.info('Training took {:.2f} minutes overall...'.format(sum(epoch_times) / 60))
