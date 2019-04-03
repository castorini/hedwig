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


class BertEvaluator(object):
    def __init(self, model, processor)
        self.args = get_args()
        self.model = model
        #self.train_dataloader = train_dataloader
        #self.optimizer = optimizer
        self.processor = processor
        self.eval_examples = None
        #self.num_train_optimization_steps = None
        self.eval_examples = self.processor.get_dev_examples(args.data_dir)
        #self.num_train_optimization_steps = int(
        #    len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        #if args.local_rank != -1:
         #   num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()  
        self.global_step = 0
        self.nb_tr_steps = 0
        self.tr_loss = 0   
        self.eval_loss, self.eval_accuracy = 0.0, 0.0
        self.nb_eval_steps, self.nb_eval_examples = 0, 0   

"""    def train_epoch(self):
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
                self.global_step += 1"""
    def evaluate(self, epochs):
        eval_features = convert_examples_to_features(
            self.eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        #logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        
        #if args.local_rank == -1:
         #   train_sampler = RandomSampler(train_data)
        #else:
         #   train_sampler = DistributedSampler(train_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()        

        for input_ids, input_mask, segment_ids, label_ids in tqdm(self.eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        self.eval_loss = self.eval_loss / self.nb_eval_steps
        self.eval_accuracy = self.eval_accuracy / self.nb_eval_examples
        #loss = tr_loss/nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step}
                  #'loss': loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
