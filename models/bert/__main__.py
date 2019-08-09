import os
import random
import time
import math

import numpy as np
import torch

from common.evaluators.bert_evaluator import BertEvaluator
from common.trainers.bert_trainer import BertTrainer
from datasets.bert_processors.aapd_processor import AAPDProcessor
from datasets.bert_processors.agnews_processor import AGNewsProcessor
from datasets.bert_processors.imdb_processor import IMDBProcessor
from datasets.bert_processors.reuters_processor import ReutersProcessor
from datasets.bert_processors.sogou_processor import SogouProcessor
from datasets.bert_processors.sst_processor import SST2Processor
from datasets.bert_processors.yelp2014_processor import Yelp2014Processor
from datasets.bert_processors.lyrics_processor import LyricsProcessor

from models.bert.args import get_args
from models.bert.model import BertForSequenceClassification
from utils.io import PYTORCH_PRETRAINED_BERT_CACHE
from utils.optimization import BertAdam
from utils.tokenization import BertTokenizer

# String templates for logging results
LOG_HEADER = 'Split  Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss'
LOG_TEMPLATE = ' '.join('{:>5s},{:>9.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))


def evaluate_split(model, processor, args, split='dev'):
    evaluator = BertEvaluator(model, processor, args, split)
    start_time = time.time()
    accuracy, precision, recall, f1, avg_loss = evaluator.get_scores(silent=True)[0]
    print("Inference time", time.time() - start_time)
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), accuracy, precision, recall, f1, avg_loss))


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()

    if args.local_rank == -1 or not args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    print('Device:', str(device).upper())
    print('Number of GPUs:', n_gpu)
    print('Distributed training:', bool(args.local_rank != -1))
    print('FP16:', args.fp16)

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    dataset_map = {
        'SST-2': SST2Processor,
        'Reuters': ReutersProcessor,
        'IMDB': IMDBProcessor,
        'AAPD': AAPDProcessor,
        'AGNews': AGNewsProcessor,
        'Yelp2014': Yelp2014Processor,
        'Sogou': SogouProcessor,
        'Lyrics': LyricsProcessor
    }

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    processor = dataset_map[args.dataset]()
    processor.set_num_classes_(args.data_dir)

    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.device = device
    args.n_gpu = n_gpu
    args.num_labels = processor.NUM_CLASSES
    args.is_multilabel = processor.IS_MULTILABEL

    if not args.trained_model:
        save_path = os.path.join(args.save_path, processor.NAME)
        os.makedirs(save_path, exist_ok=True)

    args.is_lowercase = 'uncased' in args.model
    args.is_hierarchical = False
    tokenizer = BertTokenizer.from_pretrained(args.model, is_lowercase=args.is_lowercase)

    train_examples = None
    num_train_optimization_steps = None
    if not args.trained_model:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            math.ceil(len(train_examples) / args.batch_size) / args.gradient_accumulation_steps) * args.epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassification.from_pretrained(args.model, cache_dir=cache_dir, num_labels=args.num_labels)

    if args.fp16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Install NVIDIA Apex to use distributed and FP16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install NVIDIA Apex for distributed and FP16 training")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.lr,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.lr,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)    

    trainer = BertTrainer(model, optimizer, processor, args)

    if not args.trained_model:
        trainer.train()
        model = torch.load(trainer.snapshot_path)
    else:
        model = BertForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
        model_ = torch.load(args.trained_model, map_location=lambda storage, loc: storage)
        state={}
        for key in model_.state_dict().keys():
            new_key = key.replace("module.", "")
            state[new_key] = model_.state_dict()[key]
        model.load_state_dict(state)
        model = model.to(device)

    evaluate_split(model, processor, args, split='dev')
    evaluate_split(model, processor, args, split='test')

