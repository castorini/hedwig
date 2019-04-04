import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from tqdm import trange

from datasets.processors.bert_processor import convert_examples_to_features
from utils.optimization import warmup_linear
from utils.tokenization4bert import BertTokenizer


class BertTrainer(object):
    def __init__(self, model, optimizer, processor, args):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.processor = processor
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

        self.train_examples = self.processor.get_train_examples(args.data_dir)
        self.num_train_optimization_steps = int(
            len(self.train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            self.num_train_optimization_steps = args.num_train_optimization_steps // torch.distributed.get_world_size()
        self.global_step = 0
        self.nb_tr_steps = 0
        self.tr_loss = 0      

    def train_epoch(self, train_dataloader):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits = self.model(input_ids, segment_ids, input_mask, label_ids)
            loss = F.cross_entropy(logits.view(-1, self.args.num_labels), label_ids.view(-1))
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
                    lr_this_step = self.args.learning_rate * warmup_linear(self.global_step/self.num_train_optimization_steps, self.args.warmup_proportion)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

    def train(self):
        label_list = self.processor.get_labels()
        train_features = convert_examples_to_features(
            self.train_examples, label_list, self.args.max_seq_length, self.tokenizer)
        print("***** Running training *****")
        print("  Num. of examples: ", len(self.train_examples))
        print("  Batch size:", self.args.train_batch_size)
        print("  Num of steps:", self.num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if self.args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.train_batch_size)

        self.model.train()

        for _ in trange(int(self.args.num_train_epochs), desc="Epoch"):
            self.train_epoch(train_dataloader)
