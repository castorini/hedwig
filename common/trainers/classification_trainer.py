import datetime
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.trainers.trainer import Trainer


class ClassificationTrainer(Trainer):

    def __init__(self, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator):
        super().__init__(model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)
        self.config = trainer_config
        self.early_stop = False
        self.best_dev_f1 = 0
        self.iterations = 0
        self.iters_not_improved = 0
        self.start = None
        self.log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:12.4f}'.split(','))
        self.dev_log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.4f},{:>8.4f},{:8.4f},{:12.4f},{:12.4f}'.split(','))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.snapshot_path = os.path.join(self.model_outfile, self.train_loader.dataset.NAME, '%s.pt' % timestamp)

    def train_epoch(self, epoch):
        self.train_loader.init_epoch()
        n_correct, n_total = 0, 0
        for batch_idx, batch in enumerate(self.train_loader):
            self.iterations += 1
            self.model.train()
            self.optimizer.zero_grad()
            if hasattr(self.model, 'tar') and self.model.tar:
                if 'ignore_lengths' in self.config and self.config['ignore_lengths']:
                    scores, rnn_outs = self.model(batch.text)
                else:
                    scores, rnn_outs = self.model(batch.text[0], lengths=batch.text[1])
            else:
                if 'ignore_lengths' in self.config and self.config['ignore_lengths']:
                    scores = self.model(batch.text)
                else:
                    scores = self.model(batch.text[0], lengths=batch.text[1])

            if 'is_multilabel' in self.config and self.config['is_multilabel']:
                predictions = F.sigmoid(scores).round().long()
                for tensor1, tensor2 in zip(predictions, batch.label):
                    if np.array_equal(tensor1, tensor2):
                        n_correct += 1
                loss = F.binary_cross_entropy_with_logits(scores, batch.label.float())
            else:
                for tensor1, tensor2 in zip(torch.argmax(scores, dim=1), torch.argmax(batch.label.data, dim=1)):
                    if np.array_equal(tensor1, tensor2):
                        n_correct += 1
                loss = F.cross_entropy(scores, torch.argmax(batch.label.data, dim=1))

            if hasattr(self.model, 'tar') and self.model.tar:
                loss = loss + self.model.tar * (rnn_outs[1:] - rnn_outs[:-1]).pow(2).mean()
            if hasattr(self.model, 'ar') and self.model.ar:
                loss = loss + self.model.ar * (rnn_outs[:]).pow(2).mean()

            n_total += batch.batch_size
            train_acc = 100. * n_correct / n_total
            loss.backward()
            self.optimizer.step()

            if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
                # Temporal averaging
                self.model.update_ema()

            if self.iterations % self.log_interval == 1:
                niter = epoch * len(self.train_loader) + batch_idx
                print(self.log_template.format(time.time() - self.start, epoch, self.iterations, 1 + batch_idx,
                                               len(self.train_loader), 100.0 * (1 + batch_idx) / len(self.train_loader),
                                               loss.item(), train_acc))

    def train(self, epochs):
        self.start = time.time()
        header = '  Time Epoch Iteration Progress    (%Epoch)   Loss     Accuracy'
        dev_header = '  Time Epoch Iteration Progress     Dev/Acc. Dev/Pr.  Dev/Recall   Dev/F1       Dev/Loss'
        os.makedirs(self.model_outfile, exist_ok=True)
        os.makedirs(os.path.join(self.model_outfile, self.train_loader.dataset.NAME), exist_ok=True)

        for epoch in range(1, epochs + 1):
            print('\n' + header)
            self.train_epoch(epoch)

            # Evaluate performance on validation set
            dev_acc, dev_precision, dev_recall, dev_f1, dev_loss = self.dev_evaluator.get_scores()[0]

            # Print validation results
            print('\n' + dev_header)
            print(self.dev_log_template.format(time.time() - self.start, epoch, self.iterations, epoch, epochs,
                                               dev_acc, dev_precision, dev_recall, dev_f1, dev_loss))

            # Update validation results
            if dev_f1 > self.best_dev_f1:
                self.iters_not_improved = 0
                self.best_dev_f1 = dev_f1
                torch.save(self.model, self.snapshot_path)
            else:
                self.iters_not_improved += 1
                if self.iters_not_improved >= self.patience:
                    self.early_stop = True
                    print("Early Stopping. Epoch: {}, Best Dev F1: {}".format(epoch, self.best_dev_f1))
                    break
