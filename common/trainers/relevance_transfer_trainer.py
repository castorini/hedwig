import datetime
import os

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from tqdm import trange, tqdm

from common.trainers.trainer import Trainer
from datasets.bert_processors.robust45_processor import convert_examples_to_features
from tasks.relevance_transfer.resample import ImbalancedDatasetSampler
from utils.tokenization import BertTokenizer


class RelevanceTransferTrainer(Trainer):
    def __init__(self, model, config, **kwargs):
        super().__init__(model, kwargs['embedding'], kwargs['train_loader'], config, None, kwargs['test_evaluator'], kwargs['dev_evaluator'])

        if config['model'] in {'BERT-Base', 'BERT-Large'}:
            variant = 'bert-large-uncased' if config['model'] == 'BERT-Large' else 'bert-base-uncased'
            self.tokenizer = BertTokenizer.from_pretrained(variant, is_lowercase=config['is_lowercase'])
            self.processor = kwargs['processor']
            self.optimizer = config['optimizer']
            self.train_examples = self.processor.get_train_examples(config['data_dir'], topic=config['topic'])
            self.num_train_optimization_steps = int(len(self.train_examples) /
                                                    config['batch_size'] /
                                                    config['gradient_accumulation_steps']
                                                    ) * config['epochs']
        self.config = config
        self.early_stop = False
        self.best_dev_ap = 0
        self.iterations = 0
        self.unimproved_iters = 0

        self.log_header = 'Epoch Iteration Progress   Dev/Acc.  Dev/Pr.  Dev/AP.   Dev/F1   Dev/Loss'
        self.log_template = ' '.join('{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.writer = SummaryWriter(log_dir="tensorboard_logs/" + timestamp)
        self.snapshot_path = os.path.join(self.model_outfile, config['dataset'].NAME, '%s.pt' % timestamp)

    def train_epoch(self):
        for step, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            self.model.train()

            if self.config['model'] in {'BERT-Base', 'BERT-Large'}:
                batch = tuple(t.to(self.config['device']) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = torch.sigmoid(self.model(input_ids, segment_ids, input_mask)).squeeze(dim=1)
                loss = F.binary_cross_entropy(logits, label_ids.float())

                if self.config['n_gpu'] > 1:
                    loss = loss.mean()
                if self.config['gradient_accumulation_steps'] > 1:
                    loss = loss / self.config['gradient_accumulation_steps']

                loss.backward()

                if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.iterations += 1
            else:
                # Clip gradients to address exploding gradients in LSTM
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 25.0)

                # Randomly sample equal number of positive and negative documents
                self.train_loader.init_epoch()
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
                        logits, rnn_outs = torch.sigmoid(self.model(batch_text)).squeeze(dim=1)
                    else:
                        logits, rnn_outs = torch.sigmoid(self.model(batch_text, lengths=batch_lengths)).squeeze(dim=1)
                else:
                    if 'ignore_lengths' in self.config and self.config['ignore_lengths']:
                        logits = torch.sigmoid(self.model(batch_text)).squeeze(dim=1)
                    else:
                        logits = torch.sigmoid(self.model(batch_text, lengths=batch_lengths)).squeeze(dim=1)

                loss = F.binary_cross_entropy(logits, batch_label.float())
                if hasattr(self.model, 'tar') and self.model.tar:
                    loss = loss + (rnn_outs[1:] - rnn_outs[:-1]).pow(2).mean()

                loss.backward()
                self.optimizer.step()
                self.iterations += 1
                self.optimizer.zero_grad()

                if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
                    # Temporal averaging
                    self.model.update_ema()

    def train(self, epochs):
        os.makedirs(self.model_outfile, exist_ok=True)
        os.makedirs(os.path.join(self.model_outfile, self.config['dataset'].NAME), exist_ok=True)

        if self.config['model'] in {'BERT-Base', 'BERT-Large'}:
            train_features = convert_examples_to_features(
                self.train_examples, self.config['max_seq_length'], self.tokenizer)

            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            train_sampler = RandomSampler(train_data)
            self.train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=self.config['batch_size'])

        with trange(1, epochs + 1, desc="Epoch") as t_epochs:
            for epoch in t_epochs:
                self.train_epoch()

                # Evaluate performance on validation set
                dev_acc, dev_precision, dev_ap, dev_f1, dev_loss = self.dev_evaluator.get_scores()[0]
                self.writer.add_scalar('Dev/Loss', dev_loss, epoch)
                self.writer.add_scalar('Dev/Accuracy', dev_acc, epoch)
                self.writer.add_scalar('Dev/Precision', dev_precision, epoch)
                self.writer.add_scalar('Dev/AP', dev_ap, epoch)
                tqdm.write(self.log_header)
                tqdm.write(self.log_template.format(epoch, self.iterations, epoch, epochs,
                                                    dev_acc, dev_precision, dev_ap, dev_f1, dev_loss))

                # Update validation results
                if dev_f1 > self.best_dev_ap:
                    self.unimproved_iters = 0
                    self.best_dev_ap = dev_f1
                    torch.save(self.model, self.snapshot_path)
                else:
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.patience:
                        self.early_stop = True
                        tqdm.write("Early Stopping. Epoch: {}, Best Dev F1: {}".format(epoch, self.best_dev_ap))
                        t_epochs.close()
                        break
