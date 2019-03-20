import logging
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
from models.han.model import HAN
from models.han.args import get_args


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


def evaluate_dataset(split_name, dataset_cls, model, embedding, loader, batch_size, device, single_label):
    saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, batch_size, device)
    saved_model_evaluator.single_label = single_label
    saved_model_evaluator.ignore_lengths = True
    scores, metric_names = saved_model_evaluator.get_scores()
    print('Evaluation metrics for', split_name)
    print(metric_names)
    print(scores)


if __name__ == '__main__':
    # Set default configuration in : args.py
    args = get_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print('Note: You are using GPU for training')
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cuda:
        print('Warning: You have Cuda but not use it. You are using CPU for training.')
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger = get_logger()

    dataset_map = {
        'Reuters': Reuters,
        'AAPD': AAPD,
        'IMDB': IMDB,
        'Yelp2014': Yelp2014
    }

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')
    else:
        train_iter, dev_iter, test_iter = dataset_map[args.dataset].iters(args.data_dir, args.word_vectors_file, args.word_vectors_dir, batch_size=args.batch_size, device=args.gpu, unk_init=UnknownWordVecCache.unk)

    config = deepcopy(args)
    config.dataset = train_iter.dataset
    config.target_class = train_iter.dataset.NUM_CLASSES
    config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)

    print('Dataset {}    Mode {}'.format(args.dataset, args.mode))
    print('VOCAB num',len(train_iter.dataset.TEXT_FIELD.vocab))
    print('LABEL.target_class:', train_iter.dataset.NUM_CLASSES)
    print('Train instance', len(train_iter.dataset))
    print('Dev instance', len(dev_iter.dataset))
    print('Test instance', len(test_iter.dataset))

    if args.resume_snapshot:
        if args.cuda:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
    else:
        model = HAN(config)
        if args.cuda:
            model.cuda()
            print('Shift model to GPU')

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    print(parameter)
    #optimizer = torch.optim.Adadelta(parameter, lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = torch.optim.SGD(parameter, lr = args.lr, momentum = 0.9)
    optimizer = torch.optim.Adam(parameter, lr = args.lr)
    
    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')
    else:
        train_evaluator = EvaluatorFactory.get_evaluator(dataset_map[args.dataset], model, None, train_iter, args.batch_size, args.gpu)
        test_evaluator = EvaluatorFactory.get_evaluator(dataset_map[args.dataset], model, None, test_iter, args.batch_size, args.gpu)
        dev_evaluator = EvaluatorFactory.get_evaluator(dataset_map[args.dataset], model, None, dev_iter, args.batch_size, args.gpu)
        train_evaluator.single_label = args.single_label
        test_evaluator.single_label = args.single_label
        dev_evaluator.single_label = args.single_label

    dev_evaluator.ignore_lengths = True
    test_evaluator.ignore_lengths = True
    trainer_config = {
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'log_interval': args.log_every,
        'patience': args.patience,
        'model_outfile': args.save_path,   # actually a directory, using model_outfile to conform to Trainer naming convention
        'logger': logger,
        'ignore_lengths': True,
        'single_label': args.single_label
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
    model = torch.load(trainer.snapshot_path)
    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')
    else:
        evaluate_dataset('dev', dataset_map[args.dataset], model, None, dev_iter, args.batch_size, args.gpu, args.single_label)
        evaluate_dataset('test', dataset_map[args.dataset], model, None, test_iter, args.batch_size, args.gpu, args.single_label)

    if args.onnx:
        device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')
        dummy_input = torch.zeros(args.onnx_batch_size, args.onnx_sent_len, dtype=torch.long, device=device)
        onnx_filename = 'han_{}.onnx'.format(args.mode)
        torch.onnx.export(model, dummy_input, onnx_filename)
        print('Exported model in ONNX format as {}'.format(onnx_filename))
