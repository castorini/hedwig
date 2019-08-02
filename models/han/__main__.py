import logging
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.onnx

from common.evaluate import EvaluatorFactory
from common.train import TrainerFactory
from datasets.aapd import AAPDHierarchical as AAPD
from datasets.imdb import IMDBHierarchical as IMDB
from datasets.reuters import ReutersHierarchical as Reuters
from datasets.yelp2014 import Yelp2014Hierarchical as Yelp2014
from datasets.lyrics import LyricsHierarchical as Lyrics
from models.han.args import get_args
from models.han.model import HAN


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


def evaluate_dataset(split_name, dataset_cls, model, embedding, loader, batch_size, device, is_multilabel):
    saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, batch_size, device)
    if hasattr(saved_model_evaluator, 'is_multilabel'):
        saved_model_evaluator.is_multilabel = is_multilabel
    if hasattr(saved_model_evaluator, 'ignore_lengths'):
        saved_model_evaluator.ignore_lengths = True

    scores, metric_names = saved_model_evaluator.get_scores()
    print('Evaluation metrics for', split_name)
    print(metric_names)
    print(scores)


if __name__ == '__main__':
    # Set default configuration in args.py
    # args = get_args()
    logger = get_logger()
    #
    # # Set random seed for reproducibility
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    #
    # if not args.cuda:
    #     args.gpu = -1
    # if torch.cuda.is_available() and args.cuda:
    #     print('Note: You are using GPU for training')
    #     torch.cuda.set_device('cuda:{}'.format(args.gpu))
    #     torch.cuda.manual_seed(args.seed)
    # if torch.cuda.is_available() and not args.cuda:
    #     print('Warning: Using CPU for training')

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

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    dataset_map = {
        'Reuters': Reuters,
        'AAPD': AAPD,
        'IMDB': IMDB,
        'Yelp2014': Yelp2014,
        'Lyrics': Lyrics
    }

    args.device = device

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    else:
        dataset_class = dataset_map[args.dataset]
        train_iter, dev_iter, test_iter = dataset_class.iters(args.data_dir,
                                                              args.word_vectors_file,
                                                              args.word_vectors_dir,
                                                              batch_size=args.batch_size,
                                                              device=args.device,
                                                              unk_init=UnknownWordVecCache.unk)

    config = deepcopy(args)
    config.dataset = train_iter.dataset
    config.target_class = train_iter.dataset.NUM_CLASSES
    config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)

    print('Dataset:', args.dataset)
    print('No. of target classes:', train_iter.dataset.NUM_CLASSES)
    print('No. of train instances', len(train_iter.dataset))
    print('No. of dev instances', len(dev_iter.dataset))
    print('No. of test instances', len(test_iter.dataset))

    if args.resume_snapshot:
        if args.cuda:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.device))
        else:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
    else:
        model = HAN(config)
        if args.cuda:
            model.to(device)

    if not args.trained_model:
        save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME)
        os.makedirs(save_path, exist_ok=True)

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)
    
    train_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, train_iter, args.batch_size, args.gpu)
    test_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, test_iter, args.batch_size, args.gpu)
    dev_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, dev_iter, args.batch_size, args.gpu)

    if hasattr(train_evaluator, 'is_multilabel'):
        train_evaluator.is_multilabel = dataset_class.IS_MULTILABEL
    if hasattr(dev_evaluator, 'is_multilabel'):
        dev_evaluator.is_multilabel = dataset_class.IS_MULTILABEL
    if hasattr(dev_evaluator, 'ignore_lengths'):
        dev_evaluator.ignore_lengths = True
    if hasattr(test_evaluator, 'is_multilabel'):
        test_evaluator.is_multilabel = dataset_class.IS_MULTILABEL
    if hasattr(test_evaluator, 'ignore_lengths'):
        test_evaluator.ignore_lengths = True

    trainer_config = {
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'log_interval': args.log_every,
        'patience': args.patience,
        'model_outfile': args.save_path,
        'logger': logger,
        'is_multilabel': dataset_class.IS_MULTILABEL,
        'ignore_lengths': True
    }

    trainer = TrainerFactory.get_trainer(args.dataset, model, None, train_iter, trainer_config, train_evaluator, test_evaluator, dev_evaluator)

    if not args.trained_model:
        trainer.train(args.epochs)
    else:
        if args.cuda:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage)

    # Calculate dev and test metrics
    if hasattr(trainer, 'snapshot_path'):
        model = torch.load(trainer.snapshot_path)

    evaluate_dataset('dev', dataset_map[args.dataset], model, None, dev_iter, args.batch_size,
                     is_multilabel=dataset_class.IS_MULTILABEL,
                     device=args.gpu)
    evaluate_dataset('test', dataset_map[args.dataset], model, None, test_iter, args.batch_size,
                     is_multilabel=dataset_class.IS_MULTILABEL,
                     device=args.gpu)
