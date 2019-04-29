import json
import logging
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from common.evaluate import EvaluatorFactory
from common.train import TrainerFactory
from datasets.robust04 import Robust04, Robust04Hierarchical
from datasets.robust05 import Robust05, Robust05Hierarchical
from datasets.robust45 import Robust45, Robust45Hierarchical
from models.han.model import HAN
from models.kim_cnn.model import KimCNN
from models.reg_lstm.model import RegLSTM
from models.xml_cnn.model import XmlCNN
from tasks.relevance_transfer.args import get_args
from tasks.relevance_transfer.rerank import rerank


# String templates for logging results
LOG_HEADER = 'Topic  Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss'
LOG_TEMPLATE = ' '.join('{:>5s},{:>9.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))


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


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def evaluate_dataset(split, dataset_cls, model, embedding, loader, pred_scores, args, topic):
    saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, args.batch_size, args.gpu)
    if args.model in {'HAN', 'HR-CNN'}:
        saved_model_evaluator.ignore_lengths = True
    accuracy, precision, recall, f1, avg_loss = saved_model_evaluator.get_scores()[0]

    if split == 'test':
        pred_scores[topic] = (saved_model_evaluator.y_pred, saved_model_evaluator.docid)
    else:
        print('\n' + LOG_HEADER)
        print(LOG_TEMPLATE.format(topic, accuracy, precision, recall, f1, avg_loss) + '\n')

    return saved_model_evaluator.y_pred


def save_ranks(pred_scores, output_path):
    with open(output_path, 'w') as output_file:
        for topic in tqdm(pred_scores, desc='Saving'):
            scores, docid = pred_scores[topic]
            max_scores = defaultdict(list)
            for score, docid in zip(scores, docid):
                max_scores[docid].append(score)
            sorted_score = sorted(((sum(scores)/len(scores), docid) for docid, scores in max_scores.items()), reverse=True)
            rank = 1  # Reset rank counter to one
            for score, docid in sorted_score:
                output_file.write(f'{topic} Q0 {docid} {rank} {score} Castor\n')
                rank += 1


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()
    logger = get_logger()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print('Note: You are using GPU for training')
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cuda:
        print('Warning: Using CPU for training')

    dataset_map = {
        'Robust04': Robust04,
        'Robust45': Robust45,
        'Robust05': Robust05
    }

    dataset_map_hi = {
        'Robust04': Robust04Hierarchical,
        'Robust45': Robust45Hierarchical,
        'Robust05': Robust05Hierarchical
    }

    model_map = {
        'RegLSTM': RegLSTM,
        'KimCNN': KimCNN,
        'HAN': HAN,
        'XML-CNN': XmlCNN,
    }

    if args.model in {'HAN', 'HR-CNN'}:
        dataset = dataset_map_hi[args.dataset]
    else:
        dataset = dataset_map[args.dataset]
    print('Dataset:', args.dataset)

    if args.rerank:
        rerank(args, dataset)

    else:
        topic_iter = 0
        cache_path = os.path.splitext(args.output_path)[0] + '.pkl'
        if args.resume_snapshot:
            # Load previous cached run
            with open(cache_path, 'rb') as cache_file:
                pred_scores = pickle.load(cache_file)
        else:
            pred_scores = dict()

        with open(os.path.join('tasks', 'relevance_transfer', 'config.json'), 'r') as config_file:
            topic_configs = json.load(config_file)

        for topic in dataset.TOPICS:
            topic_iter += 1
            # Skip topics that have already been predicted
            if args.resume_snapshot and topic in pred_scores:
                continue

            print("Training on topic %d of %d..." % (topic_iter, len(dataset.TOPICS)))
            train_iter, dev_iter, test_iter = dataset.iters(args.data_dir, args.word_vectors_file, args.word_vectors_dir,
                                                            topic, batch_size=args.batch_size, device=args.gpu,
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
                        print("Setting dynamic_pool_length to", topic_configs[args.model][topic]["dynamic_pool_length"])
                        config.dynamic_pool_length = topic_configs[args.model][topic]["dynamic_pool_length"]

            model = model_map[args.model](config)

            if args.cuda:
                model.cuda()
                print('Shifting model to GPU...')

            parameter = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)

            if args.dataset not in dataset_map:
                raise ValueError('Unrecognized dataset')
            else:
                train_evaluator = EvaluatorFactory.get_evaluator(dataset_map[args.dataset], model, None, train_iter,
                                                                 args.batch_size, args.gpu)
                test_evaluator = EvaluatorFactory.get_evaluator(dataset_map[args.dataset], model, None, test_iter,
                                                                args.batch_size, args.gpu)
                dev_evaluator = EvaluatorFactory.get_evaluator(dataset_map[args.dataset], model, None, dev_iter,
                                                               args.batch_size, args.gpu)

            trainer_config = {
                'optimizer': optimizer,
                'batch_size': args.batch_size,
                'log_interval': args.log_every,
                'dev_log_interval': args.dev_every,
                'patience': args.patience,
                'model_outfile': args.save_path,
                'logger': logger,
                'resample': args.resample
            }

            if args.model in {'HAN', 'HR-CNN'}:
                trainer_config['ignore_lengths'] = True
                dev_evaluator.ignore_lengths = True
                test_evaluator.ignore_lengths = True

            trainer = TrainerFactory.get_trainer(args.dataset, model, None, train_iter, trainer_config, train_evaluator,
                                                 test_evaluator, dev_evaluator)

            trainer.train(args.epochs)

            # Calculate dev and test metrics
            model = torch.load(trainer.snapshot_path)

            if hasattr(model, 'beta_ema') and model.beta_ema > 0:
                old_params = model.get_params()
                model.load_ema_params()

            if args.dataset not in dataset_map:
                raise ValueError('Unrecognized dataset')
            else:
                evaluate_dataset('dev', dataset_map[args.dataset], model, None, dev_iter, pred_scores, args, topic)
                evaluate_dataset('test', dataset_map[args.dataset], model, None, test_iter, pred_scores, args, topic)

            if hasattr(model, 'beta_ema') and model.beta_ema > 0:
                model.load_params(old_params)

            with open(cache_path, 'wb') as cache_file:
                pickle.dump(pred_scores, cache_file)

        save_ranks(pred_scores, args.output_path)
