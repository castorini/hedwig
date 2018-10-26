import time

import datetime
import numpy as np
import os
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from .trainer import Trainer


class ReutersTrainer(Trainer):

    def __init__(self, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator):
        super(ReutersTrainer, self).__init__(model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)
        self.early_stop = False
        self.best_dev_acc = 0
        self.iterations = 0
        self.iters_not_improved = 0
        self.start = None
        self.log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
        self.dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
        self.writer = SummaryWriter(log_dir="tensorboard_logs/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    def train_epoch(self, epoch):
        self.train_loader.init_epoch()
        n_correct, n_total = 0, 0
        for batch_idx, batch in enumerate(self.train_loader):
            self.iterations += 1
            self.model.train()
            self.optimizer.zero_grad()
            scores = self.model(batch.text[0], lengths=batch.text[1])
            # Using binary accuracy
            for tensor1, tensor2 in zip(F.sigmoid(scores).round().long(), batch.label):
                if np.array_equal(tensor1, tensor2):
                    n_correct += 1
            n_total += batch.batch_size
            train_acc = 100. * n_correct / n_total
            loss = F.binary_cross_entropy_with_logits(scores, batch.label.float())
            loss.backward()

            self.optimizer.step()

            # Evaluate performance on validation set
            if self.iterations % self.dev_log_interval == 1:
                dev_acc, dev_loss = self.dev_evaluator.get_scores()[0]
                niter = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.data[0], niter)
                self.writer.add_scalar('Dev/Loss', dev_loss, niter)
                self.writer.add_scalar('Train/Accuracy', train_acc, niter)
                self.writer.add_scalar('Dev/Accuracy', dev_acc, niter)
                print(self.dev_log_template.format(time.time() - self.start,
                      epoch, self.iterations, 1 + batch_idx, len(self.train_loader),
                      100. * (1 + batch_idx) / len(self.train_loader), loss.item(),
                      dev_loss, train_acc, dev_acc))

                # Update validation results
                if dev_acc > self.best_dev_acc:
                    self.iters_not_improved = 0
                    self.best_dev_acc = dev_acc
                    snapshot_path = os.path.join(self.model_outfile, self.train_loader.dataset.NAME, self.model.mode + '_best_model.pt')
                    torch.save(self.model, snapshot_path)
                else:
                    self.iters_not_improved += 1
                    if self.iters_not_improved >= self.patience:
                        self.early_stop = True
                        break

            if self.iterations % self.log_interval == 1:
                # print progress message
                print(self.log_template.format(time.time() - self.start,
                                          epoch, self.iterations, 1 + batch_idx, len(self.train_loader),
                                          100. * (1 + batch_idx) / len(self.train_loader), loss.item(), ' ' * 8,
                                          train_acc, ' ' * 12))

    def train(self, epochs):
        self.start = time.time()
        header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
        # model_outfile is actually a directory, using model_outfile to conform to Trainer naming convention
        os.makedirs(self.model_outfile, exist_ok=True)
        os.makedirs(os.path.join(self.model_outfile, self.train_loader.dataset.NAME), exist_ok=True)
        print(header)

        for epoch in range(1, epochs + 1):
            if self.early_stop:
                print("Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, self.best_dev_acc))
                break
            self.train_epoch(epoch)
