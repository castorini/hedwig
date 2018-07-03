from copy import deepcopy
import logging
import random

import numpy as np
import torch

from common.evaluation import EvaluatorFactory
from common.train import TrainerFactory
from datasets.sst import SST1
from datasets.sst import SST2
from kim_cnn.args import get_args
from kim_cnn.model import KimCNN


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def evaluate_dataset(split_name, dataset_cls, model, embedding, loader, batch_size, device):
    saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, batch_size, device)
    scores, metric_names = saved_model_evaluator.get_scores()
    logger.info('Evaluation metrics for {}'.format(split_name))
    logger.info('\t'.join([' '] + metric_names))
    logger.info('\t'.join([split_name] + list(map(str, scores))))


if __name__ == '__main__':
    # Set default configuration in : args.py
    args = get_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print("Note: You are using GPU for training")
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cuda:
        print("Warning: You have Cuda but not use it. You are using CPU for training.")
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger = get_logger()

    # Set up the data for training SST-1
    if args.dataset == 'SST-1':
        train_iter, dev_iter, test_iter = SST1.iters(args.data_dir, args.word_vectors_file, args.word_vectors_dir, batch_size=args.batch_size, device=args.gpu)
    # Set up the data for training SST-2
    elif args.dataset == 'SST-2':
        train_iter, dev_iter, test_iter = SST2.iters(args.data_dir, args.word_vectors_file, args.word_vectors_dir, batch_size=args.batch_size, device=args.gpu)
    else:
        raise ValueError('Unrecognized dataset')

    config = deepcopy(args)
    config.dataset = train_iter.dataset
    config.target_class = train_iter.dataset.NUM_CLASSES
    config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)

    print("Dataset {}    Mode {}".format(args.dataset, args.mode))
    print("VOCAB num",len(train_iter.dataset.TEXT_FIELD.vocab))
    print("LABEL.target_class:", train_iter.dataset.NUM_CLASSES)
    print("Train instance", len(train_iter.dataset))
    print("Dev instance", len(dev_iter.dataset))
    print("Test instance", len(test_iter.dataset))

    if args.resume_snapshot:
        if args.cuda:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
    else:
        model = KimCNN(config)
        if args.cuda:
            model.cuda()
            print("Shift model to GPU")

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adadelta(parameter, lr=args.lr, weight_decay=args.weight_decay)

    if args.dataset == 'SST-1':
        train_evaluator = EvaluatorFactory.get_evaluator(SST1, model, None, train_iter, args.batch_size, args.gpu)
        test_evaluator = EvaluatorFactory.get_evaluator(SST1, model, None, test_iter, args.batch_size, args.gpu)
        dev_evaluator = EvaluatorFactory.get_evaluator(SST1, model, None, dev_iter, args.batch_size, args.gpu)
    elif args.dataset == 'SST-2':
        train_evaluator = EvaluatorFactory.get_evaluator(SST2, model, None, train_iter, args.batch_size, args.gpu)
        test_evaluator = EvaluatorFactory.get_evaluator(SST2, model, None, test_iter, args.batch_size, args.gpu)
        dev_evaluator = EvaluatorFactory.get_evaluator(SST2, model, None, dev_iter, args.batch_size, args.gpu)
    else:
        raise ValueError('Unrecognized dataset')

    trainer_config = {
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'log_interval': args.log_every,
        'dev_log_interval': args.dev_every,
        'patience': args.patience,
        'model_outfile': args.save_path,   # actually a directory, using model_outfile to conform to Trainer naming convention
        'logger': logger
    }
    trainer = TrainerFactory.get_trainer(args.dataset, model, None, train_iter, trainer_config, train_evaluator, test_evaluator, dev_evaluator)

    if not args.trained_model:
        trainer.train(args.epochs)
    else:
        if args.cuda:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage)

    if args.dataset == 'SST-1':
        evaluate_dataset('dev', SST1, model, None, dev_iter, args.batch_size, args.gpu)
        evaluate_dataset('test', SST1, model, None, test_iter, args.batch_size, args.gpu)
    elif args.dataset == 'SST-2':
        evaluate_dataset('dev', SST2, model, None, dev_iter, args.batch_size, args.gpu)
        evaluate_dataset('test', SST2, model, None, test_iter, args.batch_size, args.gpu)
    else:
        raise ValueError('Unrecognized dataset')

