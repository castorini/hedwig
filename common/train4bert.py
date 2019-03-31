import time

import datetime
import numpy as np
import os
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from .trainer import Trainer


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
        self.writer = SummaryWriter(log_dir="tensorboard_logs/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.snapshot_path = os.path.join(self.model_outfile, self.train_loader.dataset.NAME, 'best_model.pt')

    """def train_epoch(self, epoch):
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

            # Temp Ave
            if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
                self.model.update_ema()

            if self.iterations % self.log_interval == 1:
                niter = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.data.item(), niter)
                self.writer.add_scalar('Train/Accuracy', train_acc, niter)
                print(self.log_template.format(time.time() - self.start, epoch, self.iterations, 1 + batch_idx,
                                               len(self.train_loader), 100.0 * (1 + batch_idx) / len(self.train_loader),
                                               loss.item(), train_acc))"""
    def train_epoch(self, epoch):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

    def train(self, epochs):
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            train_epoch()
"""    def train(self, epochs):
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
            self.writer.add_scalar('Dev/Loss', dev_loss, epoch)
            self.writer.add_scalar('Dev/Accuracy', dev_acc, epoch)
            self.writer.add_scalar('Dev/Precision', dev_precision, epoch)
            self.writer.add_scalar('Dev/Recall', dev_recall, epoch)
            self.writer.add_scalar('Dev/F-measure', dev_f1, epoch)
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
                    break"""
