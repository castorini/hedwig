import os
import random
import shutil

import numpy as np
import torch

from common.evaluators.bert_evaluator import BertEvaluator
from common.trainers.bert_trainer import BertTrainer
from datasets.processors.sst_processor import SST2Processor
from datasets.processors.reuters_processor import ReutersProcessor
from datasets.processors.imdb_processor import IMDBProcessor
from models.bert.args import get_args
from models.bert.model import BertForSequenceClassification
from utils.io import PYTORCH_PRETRAINED_BERT_CACHE
from utils.optimization import BertAdam
from utils.tokenization import BertTokenizer

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

    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    dataset_map = {
        'SST-2': SST2Processor,
        'Reuters': ReutersProcessor,
        'IMDB': IMDBProcessor
    }

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.device = device
    args.n_gpu = n_gpu
    args.num_labels = dataset_map[args.dataset].NUM_CLASSES
    args.is_multilabel = dataset_map[args.dataset].IS_MULTILABEL

    if os.path.exists(args.save_path) and os.listdir(args.save_path) and args.do_train:
        shutil.rmtree(args.save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    processor = dataset_map[args.dataset]()
    args.is_lowercase = 'uncased' in args.model
    tokenizer = BertTokenizer.from_pretrained(args.model, is_lowercase=args.is_lowercase)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassification.from_pretrained(args.model,
                                                          cache_dir=cache_dir,
                                                          num_labels=args.num_labels)
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
            raise ImportError("Please install Nvidia Apex for distributed and fp16 training")

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
    test_evaluator = BertEvaluator(model, processor, args, split='test')

    if args.do_train:
        trainer.train()
    else:
        model = BertForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)

    test_evaluator.evaluate()
