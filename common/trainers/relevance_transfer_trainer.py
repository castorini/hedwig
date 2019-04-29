import datetime
import os
import time

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm

from common.trainers.trainer import Trainer
from tasks.relevance_transfer.resample import ImbalancedDatasetSampler


class RelevanceTransferTrainer(Trainer):

    def __init__(self, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator):
        super(RelevanceTransferTrainer, self).__init__(model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)
        self.config = trainer_config
        self.early_stop = False
        self.best_dev_ap = 0
        self.iterations = 0
        self.iters_not_improved = 0
        self.start = None

        self.log_header = 'Epoch Iteration Progress   Dev/Acc.  Dev/Pr.  Dev/AP.   Dev/F1   Dev/Loss'
        self.log_template = ' '.join('{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.writer = SummaryWriter(log_dir="tensorboard_logs/" + timestamp)
        self.snapshot_path = os.path.join(self.model_outfile, self.train_loader.dataset.NAME, '%s.pt' % timestamp)

    def train_epoch(self, epoch):
        self.train_loader.init_epoch()
        n_correct, n_total = 0, 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            self.iterations += 1
            self.model.train()
            self.optimizer.zero_grad()

            # Clip gradients to address exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 25.0)

            # Randomly sample equal number of positive and negative documents
            if 'ignore_lengths' in self.config and self.config['ignore_lengths']:
                if 'resample' in self.config and self.config['resample']:
                    indices = ImbalancedDatasetSampler(batch.text, batch.label).get_indices()
                    batch_text = batch.text[indices]
                    batch_label = batch.label[indices]
                else:
                    batch_text = batch.text
                    batch_label = batch.label
            else:
                if 'resample' in self.config and self.config['resample']:
                    indices = ImbalancedDatasetSampler(batch.text[0], batch.label).get_indices()
                    batch_text = batch.text[0][indices]
                    batch_lengths = batch.text[1][indices]
                    batch_label = batch.label
                else:
                    batch_text = batch.text[0]
                    batch_lengths = batch.text[1]
                    batch_label = batch.label

            if hasattr(self.model, 'tar') and self.model.tar:
                if 'ignore_lengths' in self.config and self.config['ignore_lengths']:
                    scores, rnn_outs = self.model(batch_text)
                else:
                    scores, rnn_outs = self.model(batch_text, lengths=batch_lengths)
            else:
                if 'ignore_lengths' in self.config and self.config['ignore_lengths']:
                    scores = self.model(batch_text)
                else:
                    scores = self.model(batch_text, lengths=batch_lengths)

            # Computing accuracy and loss
            predictions = torch.sigmoid(scores).squeeze(dim=1)
            for tensor1, tensor2 in zip(predictions.round(), batch_label):
                try:
                    if int(tensor1.item()) == int(tensor2.item()):
                        n_correct += 1
                except ValueError:
                    # Ignore NaN/Inf values
                    pass

            loss = F.binary_cross_entropy(predictions, batch_label.float())

            if hasattr(self.model, 'tar') and self.model.tar:
                loss = loss + (rnn_outs[1:] - rnn_outs[:-1]).pow(2).mean()

            n_total += batch.batch_size
            train_acc = n_correct / n_total
            loss.backward()
            self.optimizer.step()

            if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
                # Temporal averaging
                self.model.update_ema()

            if self.iterations % self.log_interval == 1:
                niter = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.data.item(), niter)
                self.writer.add_scalar('Train/Accuracy', train_acc, niter)

    def train(self, epochs):
        self.start = time.time()
        # model_outfile is actually a directory, using model_outfile to conform to Trainer naming convention
        os.makedirs(self.model_outfile, exist_ok=True)
        os.makedirs(os.path.join(self.model_outfile, self.train_loader.dataset.NAME), exist_ok=True)

        for epoch in trange(1, epochs + 1, desc="Epoch"):
            self.train_epoch(epoch)

            # Evaluate performance on validation set
            dev_acc, dev_precision, dev_ap, dev_f1, dev_loss = self.dev_evaluator.get_scores()[0]
            self.writer.add_scalar('Dev/Loss', dev_loss, epoch)
            self.writer.add_scalar('Dev/Accuracy', dev_acc, epoch)
            self.writer.add_scalar('Dev/Precision', dev_precision, epoch)
            self.writer.add_scalar('Dev/AP', dev_ap, epoch)
            tqdm.write(self.log_header)
            tqdm.write(self.log_template.format(epoch, self.iterations, epoch + 1, epochs,
                                                dev_acc, dev_precision, dev_ap, dev_f1, dev_loss))

            # Update validation results
            if dev_f1 > self.best_dev_ap:
                self.iters_not_improved = 0
                self.best_dev_ap = dev_f1
                torch.save(self.model, self.snapshot_path)
            else:
                self.iters_not_improved += 1
                if self.iters_not_improved >= self.patience:
                    self.early_stop = True
                    tqdm.write("Early Stopping. Epoch: {}, Best Dev F1: {}".format(epoch, self.best_dev_ap))
                    break
