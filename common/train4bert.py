import time

import datetime
import numpy as np
import os
import torch
import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from .trainer import Trainer
from models.bert.args import get_args
from utils.optimization import warmup_linear
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)


class BertTrainer(object):
    def __init(self, model, optimizer, processor)
        self.args = get_args()
        self.model = model
        #self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.processor = processor
        self.train_examples = None
        self.num_train_optimization_steps = None
        self.train_examples = self.processor.get_train_examples(args.data_dir)
        self.num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()  
        self.global_step = 0
        self.nb_tr_steps = 0
        self.tr_loss = 0      

    def train_epoch(self):
        for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = self.model(input_ids, segment_ids, input_mask, label_ids) #model no more returns the loss, change this
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            if self.args.fp16:
                self.optimizer.backward(loss)
            else:
                loss.backward()

            self.tr_loss += loss.item()
            self.nb_tr_examples += input_ids.size(0)
            self.nb_tr_steps += 1
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                if self.args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = self.args.learning_rate * warmup_linear(self.global_step/self.num_train_optimization_steps, self.args.warmup_proportion)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

    def train(self, epochs):
        train_features = convert_examples_to_features(
            self.train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()        
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.train_epoch()
