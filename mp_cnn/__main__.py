import argparse
import logging
import os
import pprint
import random

import numpy as np
import torch
import torch.optim as optim

from common.dataset import DatasetFactory
from common.evaluation import EvaluatorFactory
from common.train import TrainerFactory
from utils.serialization import load_checkpoint
from .model import MPCNN
from .lite_model import MPCNNLite


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
    parser = argparse.ArgumentParser(description='PyTorch implementation of Multi-Perspective CNN')
    parser.add_argument('model_outfile', help='file to save final model')
    parser.add_argument('--arch', help='model architecture to use', choices=['mpcnn', 'mpcnn_lite'], default='mpcnn')
    parser.add_argument('--dataset', help='dataset to use, one of [sick, msrvid, trecqa, wikiqa]', default='sick')
    parser.add_argument('--word-vectors-dir', help='word vectors directory',
                        default=os.path.join(os.pardir, 'Castor-data', 'embeddings', 'GloVe'))
    parser.add_argument('--word-vectors-file', help='word vectors filename', default='glove.840B.300d.txt')
    parser.add_argument('--word-vectors-dim', type=int, default=300,
                        help='number of dimensions of word vectors (default: 300)')
    parser.add_argument('--skip-training', help='will load pre-trained model', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='GPU device, -1 for CPU (default: 0)')
    parser.add_argument('--wide-conv', action='store_true', default=False,
                        help='use wide convolution instead of narrow convolution (default: false)')
    parser.add_argument('--attention', choices=['none', 'basic', 'idf'], default='none', help='type of attention to use')
    parser.add_argument('--sparse-features', action='store_true',
                        default=False, help='use sparse features (default: false)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use: adam or sgd (default: adam)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--lr-reduce-factor', type=float, default=0.3,
                        help='learning rate reduce factor after plateau (default: 0.3)')
    parser.add_argument('--patience', type=float, default=2,
                        help='learning rate patience after seeing plateau (default: 2)')
    parser.add_argument('--momentum', type=float, default=0, help='momentum (default: 0)')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Optimizer epsilon (default: 1e-8)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--regularization', type=float, default=0.0001,
                        help='Regularization for the optimizer (default: 0.0001)')
    parser.add_argument('--max-window-size', type=int, default=3,
                        help='windows sizes will be [1,max_window_size] and infinity (default: 3)')
    parser.add_argument('--holistic-filters', type=int, default=300, help='number of holistic filters (default: 300)')
    parser.add_argument('--per-dim-filters', type=int, default=20, help='number of per-dimension filters (default: 20)')
    parser.add_argument('--hidden-units', type=int, default=150,
                        help='number of hidden units in each of the two hidden layers (default: 150)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='use TensorBoard to visualize training (default: false)')
    parser.add_argument('--run-label', type=str, help='label to describe run')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != -1:
        torch.cuda.manual_seed(args.seed)

    logger = get_logger()
    logger.info(pprint.pformat(vars(args)))

    dataset_cls, embedding, train_loader, test_loader, dev_loader \
        = DatasetFactory.get_dataset(args.dataset, args.word_vectors_dir, args.word_vectors_file, args.batch_size, args.device)

    filter_widths = list(range(1, args.max_window_size + 1)) + [np.inf]
    ext_feats = dataset_cls.EXT_FEATS if args.sparse_features else 0

    model_cls = MPCNN if args.arch == 'mpcnn' else MPCNNLite
    model = model_cls(args.word_vectors_dim, args.holistic_filters, args.per_dim_filters, filter_widths,
                  args.hidden_units, dataset_cls.NUM_CLASSES, args.dropout, ext_feats,
                  args.attention, args.wide_conv)

    model = model.to(device)
    embedding = embedding.to(device)

    optimizer = None
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.regularization, eps=args.epsilon)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.regularization)
    else:
        raise ValueError('optimizer not recognized: it should be either adam or sgd')

    train_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, train_loader, args.batch_size, args.device)
    test_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, test_loader, args.batch_size, args.device)
    dev_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, dev_loader, args.batch_size, args.device)

    trainer_config = {
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'log_interval': args.log_interval,
        'model_outfile': args.model_outfile,
        'lr_reduce_factor': args.lr_reduce_factor,
        'patience': args.patience,
        'tensorboard': args.tensorboard,
        'run_label': args.run_label,
        'logger': logger
    }
    trainer = TrainerFactory.get_trainer(args.dataset, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)

    if not args.skip_training:
        total_params = 0
        for param in model.parameters():
            size = [s for s in param.size()]
            total_params += np.prod(size)
        logger.info('Total number of parameters: %s', total_params)
        trainer.train(args.epochs)

    _, _, state_dict, _, _ = load_checkpoint(args.model_outfile)

    for k, tensor in state_dict.items():
        state_dict[k] = tensor.to(device)

    model.load_state_dict(state_dict)
    if dev_loader:
        evaluate_dataset('dev', dataset_cls, model, embedding, dev_loader, args.batch_size, args.device)
    evaluate_dataset('test', dataset_cls, model, embedding, test_loader, args.batch_size, args.device)
