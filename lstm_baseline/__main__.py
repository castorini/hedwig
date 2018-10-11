from copy import deepcopy
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from common.evaluation import EvaluatorFactory
from common.train import TrainerFactory
from datasets.sst import SST1
from datasets.sst import SST2
from datasets.reuters import Reuters
from lstm_baseline.args import get_args
from lstm_baseline.model import LSTMBaseline


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
            # choose 0.25 so unknown vectors have approximately same variance as pre-trained ones
            # same as original implementation: https://github.com/yoonkim/CNN_sentence/blob/0a626a048757d5272a7e8ccede256a434a6529be/process_data.py#L95
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
        print('Note: You are using GPU for training')
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cuda:
        print('Warning: You have Cuda but not use it. You are using CPU for training.')
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger = get_logger()

    # Set up the data for training SST-1
    if args.dataset == 'SST-1':
        train_iter, dev_iter, test_iter = SST1.iters(args.data_dir, args.word_vectors_file, args.word_vectors_dir, batch_size=args.batch_size, device=args.gpu, unk_init=UnknownWordVecCache.unk)
    # Set up the data for training SST-2
    elif args.dataset == 'SST-2':
        train_iter, dev_iter, test_iter = SST2.iters(args.data_dir, args.word_vectors_file, args.word_vectors_dir, batch_size=args.batch_size, device=args.gpu, unk_init=UnknownWordVecCache.unk)
    elif args.dataset == 'Reuters':
        train_iter, dev_iter, test_iter = Reuters.iters(args.data_dir, args.word_vectors_file, args.word_vectors_dir, batch_size=args.batch_size, device=args.gpu, unk_init=UnknownWordVecCache.unk)
    else:
        raise ValueError('Unrecognized dataset')

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
        model = LSTMBaseline(config)
        if args.cuda:
            model.cuda()
            print('Shift model to GPU')

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)

    if args.dataset == 'SST-1':
        train_evaluator = EvaluatorFactory.get_evaluator(SST1, model, None, train_iter, args.batch_size, args.gpu)
        test_evaluator = EvaluatorFactory.get_evaluator(SST1, model, None, test_iter, args.batch_size, args.gpu)
        dev_evaluator = EvaluatorFactory.get_evaluator(SST1, model, None, dev_iter, args.batch_size, args.gpu)
    elif args.dataset == 'SST-2':
        train_evaluator = EvaluatorFactory.get_evaluator(SST2, model, None, train_iter, args.batch_size, args.gpu)
        test_evaluator = EvaluatorFactory.get_evaluator(SST2, model, None, test_iter, args.batch_size, args.gpu)
        dev_evaluator = EvaluatorFactory.get_evaluator(SST2, model, None, dev_iter, args.batch_size, args.gpu)
    elif args.dataset == 'Reuters':
        train_evaluator = EvaluatorFactory.get_evaluator(Reuters, model, None, train_iter, args.batch_size, args.gpu)
        test_evaluator = EvaluatorFactory.get_evaluator(Reuters, model, None, test_iter, args.batch_size, args.gpu)
        dev_evaluator = EvaluatorFactory.get_evaluator(Reuters, model, None, dev_iter, args.batch_size, args.gpu)
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
    elif args.dataset == 'Reuters':
        evaluate_dataset('dev', Reuters, model, None, dev_iter, args.batch_size, args.gpu)
        evaluate_dataset('test', Reuters, model, None, test_iter, args.batch_size, args.gpu)
    else:
        raise ValueError('Unrecognized dataset')

    # Calculate dev and test metrics
    for data_loader in [dev_iter, test_iter]:
        predicted_labels = list()
        target_labels = list()
        for batch_idx, batch in enumerate(data_loader):
            scores_rounded = F.sigmoid(model(batch.text)).round().long()
            predicted_labels.extend(scores_rounded.cpu().detach().numpy())
            target_labels.extend(batch.label.cpu().detach().numpy())
        predicted_labels = np.array(predicted_labels)
        target_labels = np.array(target_labels)
        accuracy = metrics.accuracy_score(target_labels, predicted_labels)
        precision = metrics.precision_score(target_labels, predicted_labels, average='micro')
        recall = metrics.recall_score(target_labels, predicted_labels, average='micro')
        f1 = metrics.f1_score(target_labels, predicted_labels, average='micro')
        if data_loader == dev_iter:
            print("Dev metrics:")
        else:
            print("Test metrics:")
        print(accuracy, precision, recall, f1)
