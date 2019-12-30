import json
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification as Bert, AdamW, WarmupLinearSchedule, BertTokenizer

from common.constants import BERT_MODELS, PRETRAINED_MODEL_ARCHIVE_MAP, PRETRAINED_VOCAB_ARCHIVE_MAP
from common.constants import LOG_HEADER, LOG_TEMPLATE
from common.evaluators.relevance_transfer_evaluator import RelevanceTransferEvaluator
from common.trainers.relevance_transfer_trainer import RelevanceTransferTrainer
from datasets.bert_processors.robust45_processor import Robust45Processor
from datasets.robust04 import Robust04, Robust04Hierarchical
from datasets.robust05 import Robust05, Robust05Hierarchical
from datasets.robust45 import Robust45, Robust45Hierarchical
from models.han.model import HAN
from models.kim_cnn.model import KimCNN
from models.reg_lstm.model import RegLSTM
from models.xml_cnn.model import XmlCNN
from tasks.relevance_transfer.args import get_args
from tasks.relevance_transfer.rerank import rerank


class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].uniform_(-0.25, 0.25)
        return cls.cache[size_tup]


def evaluate_split(model, topic, split, config, **kwargs):
    evaluator_config = {
        'model': config.model,
        'topic': topic,
        'split': split,
        'dataset': kwargs['dataset'],
        'batch_size': config.batch_size,
        'ignore_lengths': False,
        'is_lowercase': True,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'max_seq_length': config.max_seq_length,
        'max_doc_length': args.max_doc_length,
        'data_dir': config.data_dir,
        'n_gpu': n_gpu,
        'device': config.device
    }

    if config.model in {'HAN', 'HR-CNN'}:
        trainer_config['ignore_lengths'] = True
        evaluator_config['ignore_lengths'] = True

    evaluator = RelevanceTransferEvaluator(model, evaluator_config,
                                           processor=kwargs['processor'],
                                           tokenizer=kwargs['tokenizer'],
                                           data_loader=kwargs['loader'],
                                           dataset=kwargs['dataset'])

    accuracy, precision, recall, f1, avg_loss = evaluator.get_scores()[0]

    if split == 'test':
        pred_scores[topic] = (evaluator.y_pred, evaluator.docid)
    else:
        print('\n' + LOG_HEADER)
        print(LOG_TEMPLATE.format(topic, accuracy, precision, recall, f1, avg_loss) + '\n')

    return evaluator.y_pred


def save_ranks(pred_scores, output_path):
    with open(output_path, 'w') as output_file:
        for topic in tqdm(pred_scores, desc='Saving'):
            scores, docid = pred_scores[topic]
            max_scores = defaultdict(list)
            for score, docid in zip(scores, docid):
                max_scores[docid].append(score)
            sorted_score = sorted(((sum(scores) / len(scores), docid) for docid, scores in max_scores.items()),
                                  reverse=True)
            rank = 1  # Reset rank counter to one
            for score, docid in sorted_score:
                output_file.write(f'{topic} Q0 {docid} {rank} {score} Castor\n')
                rank += 1


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()

    if torch.cuda.is_available() and not args.cuda:
        print('Warning: Using CPU for training')

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device = device
    args.n_gpu = n_gpu
    args.num_labels = 1

    print('Device:', str(device).upper())
    print('Number of GPUs:', n_gpu)

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_map = {
        'Robust04': Robust04,
        'Robust45': Robust45,
        'Robust05': Robust05
    }

    dataset_map_hier = {
        'Robust04': Robust04Hierarchical,
        'Robust45': Robust45Hierarchical,
        'Robust05': Robust05Hierarchical
    }

    dataset_map_bert = {
        'Robust45': Robust45Processor,
        'Robust04': None,
        'Robust05': None
    }

    model_map = {
        'RegLSTM': RegLSTM,
        'KimCNN': KimCNN,
        'HAN': HAN,
        'XML-CNN': XmlCNN,
        'BERT-Base': Bert,
        'BERT-Large': Bert
    }

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')
    else:
        print('Dataset:', args.dataset)

        if args.model in {'HAN', 'HR-CNN'}:
            dataset = dataset_map_hier[args.dataset]
        elif args.model in BERT_MODELS:
            dataset = dataset_map_bert[args.dataset]
        else:
            dataset = dataset_map[args.dataset]

    if args.rerank:
        rerank(args, dataset)
        exit(0)

    topic_iter = 0
    cache_path = os.path.splitext(args.output_path)[0] + '.pkl'
    save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME)
    os.makedirs(save_path, exist_ok=True)

    if args.resume_snapshot:
        # Load previous cached run
        with open(cache_path, 'rb') as cache_file:
            pred_scores = pickle.load(cache_file)
    else:
        pred_scores = dict()

    if args.model in BERT_MODELS:
        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter:", args.gradient_accumulation_steps)

        processor = dataset_map_bert[args.dataset]()
        args.is_lowercase = 'uncased' in args.model
        variant = 'bert-large-uncased' if args.model == 'BERT-Large' else 'bert-base-uncased'

        for topic in dataset.TOPICS:
            topic_iter += 1
            # Skip topics that have already been predicted
            if args.resume_snapshot and topic in pred_scores:
                continue

            print("Training on topic %d of %d..." % (topic_iter, len(dataset.TOPICS)))
            args.batch_size = args.batch_size // args.gradient_accumulation_steps
            train_examples = processor.get_train_examples(args.data_dir, topic=topic)
            num_train_optimization_steps = int(
                len(train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs

            if args.model in BERT_MODELS:
                pretrained_model_path = PRETRAINED_MODEL_ARCHIVE_MAP[variant]
                model = model_map[args.model].from_pretrained(pretrained_model_path, num_labels=1)
            else:
                model = model_map[args.model](args)

            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)

            # Prepare optimizer
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if
                            'sentence_encoder' not in n],
                 'lr': args.lr * args.lr_mult, 'weight_decay': 0.0},
                {'params': [p for n, p in param_optimizer if
                            'sentence_encoder' in n and not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if
                            'sentence_encoder' in n and any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}]

            pretrained_vocab_path = PRETRAINED_VOCAB_ARCHIVE_MAP[variant]
            tokenizer = BertTokenizer.from_pretrained(pretrained_vocab_path)
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=0.01, correct_bias=False)
            scheduler = WarmupLinearSchedule(optimizer, t_total=num_train_optimization_steps,
                                             warmup_steps=args.warmup_proportion * num_train_optimization_steps)

            trainer_config = {
                'model': args.model,
                'topic': topic,
                'dataset': dataset,
                'batch_size': args.batch_size,
                'patience': args.patience,
                'epochs': args.epochs,
                'is_lowercase': True,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'max_seq_length': args.max_seq_length,
                'max_doc_length': args.max_doc_length,
                'data_dir': args.data_dir,
                'save_path': args.save_path,
                'n_gpu': n_gpu,
                'device': args.device
            }

            evaluator_config = {
                'model': args.model,
                'topic': topic,
                'dataset': dataset,
                'split': 'dev',
                'batch_size': args.batch_size,
                'ignore_lengths': True,
                'is_lowercase': True,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'max_seq_length': args.max_seq_length,
                'max_doc_length': args.max_doc_length,
                'data_dir': args.data_dir,
                'n_gpu': n_gpu,
                'device': args.device
            }

            dev_evaluator = RelevanceTransferEvaluator(model, evaluator_config, dataset=dataset, processor=processor,
                                                       tokenizer=tokenizer)
            trainer = RelevanceTransferTrainer(model, trainer_config, optimizer=optimizer, processor=processor,
                                               tokenizer=tokenizer, scheduler=scheduler, dev_evaluator=dev_evaluator)

            trainer.train(args.epochs)
            model = torch.load(trainer.snapshot_path)

            # Calculate dev and test metrics
            evaluate_split(model, topic, 'dev', args, dataset=dataset, processor=processor, tokenizer=tokenizer, loader=None)
            evaluate_split(model, topic, 'test', args, dataset=dataset, processor=processor, tokenizer=tokenizer, loader=None)

            with open(cache_path, 'wb') as cache_file:
                pickle.dump(pred_scores, cache_file)

    else:
        if not args.cuda:
            args.gpu = -1
        if torch.cuda.is_available() and args.cuda:
            torch.cuda.set_device(args.gpu)
            torch.cuda.manual_seed(args.seed)

        with open(os.path.join('tasks', 'relevance_transfer', 'config.json'), 'r') as config_file:
            topic_configs = json.load(config_file)

        for topic in dataset.TOPICS:
            topic_iter += 1
            # Skip topics that have already been predicted
            if args.resume_snapshot and topic in pred_scores:
                continue

            print("Training on topic %d of %d..." % (topic_iter, len(dataset.TOPICS)))
            train_iter, dev_iter, test_iter = dataset.iters(args.data_dir, args.word_vectors_file,
                                                            args.word_vectors_dir, topic,
                                                            batch_size=args.batch_size, device=device,
                                                            unk_init=UnknownWordVecCache.unk)

            print('Vocabulary size:', len(train_iter.dataset.TEXT_FIELD.vocab))
            print('Target Classes:', train_iter.dataset.NUM_CLASSES)
            print('Train Instances:', len(train_iter.dataset))
            print('Dev Instances:', len(dev_iter.dataset))
            print('Test Instances:', len(test_iter.dataset))

            config = deepcopy(args)
            config.target_class = 1
            config.dataset = train_iter.dataset
            config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)

            if args.variable_dynamic_pool:
                # Set dynamic pool length based on topic configs
                if args.model in topic_configs and topic in topic_configs[args.model]:
                    print("Setting dynamic_pool to", topic_configs[args.model][topic]["dynamic_pool"])
                    config.dynamic_pool = topic_configs[args.model][topic]["dynamic_pool"]
                    if config.dynamic_pool:
                        print("Setting dynamic_pool_length to",
                              topic_configs[args.model][topic]["dynamic_pool_length"])
                        config.dynamic_pool_length = topic_configs[args.model][topic]["dynamic_pool_length"]

            model = model_map[args.model](config)

            if args.cuda:
                model.cuda()
                print('Shifting model to GPU...')

            parameter = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)

            trainer_config = {
                'model': args.model,
                'dataset': dataset,
                'batch_size': args.batch_size,
                'patience': args.patience,
                'resample': args.resample,
                'epochs': args.epochs,
                'is_lowercase': True,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'data_dir': args.data_dir,
                'save_path': args.save_path,
                'device': args.gpu
            }

            evaluator_config = {
                'topic': topic,
                'model': args.model,
                'dataset': dataset,
                'batch_size': args.batch_size,
                'ignore_lengths': False,
                'data_dir': args.data_dir,
                'device': args.gpu
            }

            if args.model in {'HAN', 'HR-CNN'}:
                trainer_config['ignore_lengths'] = True
                evaluator_config['ignore_lengths'] = True

            test_evaluator = RelevanceTransferEvaluator(model, evaluator_config, dataset=dataset, data_loader=test_iter)
            dev_evaluator = RelevanceTransferEvaluator(model, evaluator_config, dataset=dataset, data_loader=dev_iter)
            trainer = RelevanceTransferTrainer(model, trainer_config, train_loader=train_iter, optimizer=optimizer,
                                               test_evaluator=test_evaluator, dev_evaluator=dev_evaluator)

            trainer.train(args.epochs)
            model = torch.load(trainer.snapshot_path)

            if hasattr(model, 'beta_ema') and model.beta_ema > 0:
                old_params = model.get_params()
                model.load_ema_params()

            # Calculate dev and test metrics model, topic, split, config
            evaluate_split(model, topic, 'dev', args, dataset=dataset, loader=dev_iter, processor=None, tokenizer=None)
            evaluate_split(model, topic, 'test', args, dataset=dataset, loader=test_iter, processor=None, tokenizer=None)

            if hasattr(model, 'beta_ema') and model.beta_ema > 0:
                model.load_params(old_params)

            with open(cache_path, 'wb') as cache_file:
                pickle.dump(pred_scores, cache_file)

    save_ranks(pred_scores, args.output_path)
