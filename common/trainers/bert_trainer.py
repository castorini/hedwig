import datetime
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm
from tqdm import trange

from common.evaluators.bert_evaluator import BertEvaluator
from datasets.bert_processors.abstract_processor import convert_examples_to_features
from datasets.bert_processors.abstract_processor import convert_examples_to_hierarchical_features
from utils.optimization import warmup_linear
from utils.preprocessing import pad_input_matrix


class BertTrainer(object):
    def __init__(self, model, optimizer, processor, scheduler, tokenizer, args):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.processor = processor
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.train_examples = self.processor.get_train_examples(args.data_dir)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.snapshot_path = os.path.join(self.args.save_path, self.processor.NAME, '%s.pt' % timestamp)

        self.num_train_optimization_steps = int(
            len(self.train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs

        self.log_header = 'Epoch Iteration Progress   Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss'
        self.log_template = ' '.join('{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))

        self.iterations, self.nb_tr_steps, self.tr_loss = 0, 0, 0
        self.best_dev_f1, self.unimproved_iters = 0, 0
        self.early_stop = False

    def train_epoch(self, train_dataloader):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.model.train()
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits = self.model(input_ids, input_mask, segment_ids)[0]

            if self.args.is_multilabel:
                loss = F.binary_cross_entropy_with_logits(logits, label_ids.float())
            else:
                loss = F.cross_entropy(logits, torch.argmax(label_ids, dim=1))

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            if self.args.fp16:
                self.optimizer.backward(loss)
            else:
                loss.backward()

            self.tr_loss += loss.item()
            self.nb_tr_steps += 1
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                if self.args.fp16:
                    lr_this_step = self.args.learning_rate * warmup_linear(self.iterations / self.num_train_optimization_steps, self.args.warmup_proportion)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.iterations += 1

    def train(self):
        if self.args.is_hierarchical:
            train_features = convert_examples_to_hierarchical_features(
                self.train_examples, self.args.max_seq_length, self.tokenizer)
        else:
            train_features = convert_examples_to_features(
                self.train_examples, self.args.max_seq_length, self.tokenizer)

        unpadded_input_ids = [f.input_ids for f in train_features]
        unpadded_input_mask = [f.input_mask for f in train_features]
        unpadded_segment_ids = [f.segment_ids for f in train_features]

        if self.args.is_hierarchical:
            pad_input_matrix(unpadded_input_ids, self.args.max_doc_length)
            pad_input_matrix(unpadded_input_mask, self.args.max_doc_length)
            pad_input_matrix(unpadded_segment_ids, self.args.max_doc_length)

        print("Number of examples: ", len(self.train_examples))
        print("Batch size:", self.args.batch_size)
        print("Num of steps:", self.num_train_optimization_steps)

        padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
        padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
        padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids, label_ids)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.batch_size)

        for epoch in trange(int(self.args.epochs), desc="Epoch"):
            self.train_epoch(train_dataloader)
            dev_evaluator = BertEvaluator(self.model, self.processor, self.tokenizer, self.args, split='dev')
            dev_acc, dev_precision, dev_recall, dev_f1, dev_loss = dev_evaluator.get_scores()[0]

            # Print validation results
            tqdm.write(self.log_header)
            tqdm.write(self.log_template.format(epoch + 1, self.iterations, epoch + 1, self.args.epochs,
                                                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss))

            # Update validation results
            if dev_f1 > self.best_dev_f1:
                self.unimproved_iters = 0
                self.best_dev_f1 = dev_f1
                torch.save(self.model, self.snapshot_path)

            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.args.patience:
                    self.early_stop = True
                    tqdm.write("Early Stopping. Epoch: {}, Best Dev F1: {}".format(epoch, self.best_dev_f1))
                    break
