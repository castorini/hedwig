import argparse
import math
import os
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from dataset import DatasetType, MPCNNDatasetFactory
from evaluation import MPCNNEvaluatorFactory
from model import MPCNN
from train import MPCNNTrainerFactory

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of Multi-Perspective CNN')
    parser.add_argument('model_outfile', help='file to save final model')
    parser.add_argument('--dataset', help='dataset to use, one of [sick, msrvid]', default='sick')
    parser.add_argument('--word-vectors-file', help='word vectors file', default=os.path.join(os.pardir, os.pardir, 'data', 'GloVe', 'glove.840B.300d.txt'))
    parser.add_argument('--skip-training', help='will load pre-trained model', action='store_true')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--epsilon', type=float, default=1e-8, metavar='M', help='Adam epsilon (default: 1e-8)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--sample', type=int, default=0, metavar='N', help='how many examples to take from each dataset, meant for quickly testing entire end-to-end pipeline (default: all)')
    parser.add_argument('--regularization', type=float, default=0.0001, metavar='REG', help='Regularization for the optimizer (default: 0.0001)')
    parser.add_argument('--max-window-size', type=int, default=3, metavar='N', help='windows sizes will be [1,max_window_size] and infinity')
    parser.add_argument('--holistic-filters', type=int, default=300, metavar='N', help='number of holistic filters')
    parser.add_argument('--per-dim-filters', type=int, default=20, metavar='N', help='number of per-dimension filters')
    parser.add_argument('--hidden-units', type=int, default=150, metavar='N', help='number of hidden units in each of the two hidden layers')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader, test_loader, dev_loader = MPCNNDatasetFactory.get_dataset(args.dataset, args.word_vectors_file, args.batch_size, args.cuda, args.sample)

    filter_widths = list(range(1, args.max_window_size + 1)) + [np.inf]
    model = MPCNN(300, args.holistic_filters, args.per_dim_filters, filter_widths, args.hidden_units, train_loader.dataset.num_classes)
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.regularization, eps=args.epsilon)
    train_evaluator = MPCNNEvaluatorFactory.get_evaluator(args.dataset, model, train_loader, args.batch_size, args.cuda)
    test_evaluator = MPCNNEvaluatorFactory.get_evaluator(args.dataset, model, test_loader, args.batch_size, args.cuda)
    dev_evaluator = MPCNNEvaluatorFactory.get_evaluator(args.dataset, model, dev_loader, args.batch_size, args.cuda)

    trainer = MPCNNTrainerFactory.get_trainer(args.dataset, model, optimizer, train_loader, args.batch_size, args.sample, args.log_interval, args.model_outfile, train_evaluator, test_evaluator, dev_evaluator)

    if not args.skip_training:
        total_params = 0
        for param in model.parameters():
            size = [s for s in param.size()]
            total_params += np.prod(size)
        logger.info('Total number of parameters: %s', total_params)
        trainer.train(args.epochs)

    model = torch.load(args.model_outfile)
    test_evaluator = MPCNNEvaluatorFactory.get_evaluator(args.dataset, model, test_loader, args.batch_size, args.cuda)
    scores, metric_names = test_evaluator.get_scores()
    logger.info('Evaluation metrics for test')
    logger.info('\t'.join([' '] + metric_names))
    logger.info('\t'.join(['test'] + list(map(str, scores))))
