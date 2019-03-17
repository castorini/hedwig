from copy import deepcopy
import logging
import random

import numpy as np
import torch
import torch.onnx

from common.evaluation import EvaluatorFactory
from common.train import TrainerFactory
from datasets.sst import SST1
from datasets.sst import SST2
from datasets.reuters import Reuters
from datasets.aapd import AAPD
from datasets.yelp2014 import Yelp2014
from datasets.imdb import IMDB
from models.xml_cnn.args import get_args
from models.xml_cnn.model import XmlCNN


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


def evaluate_dataset(split_name, dataset_cls, model, embedding, loader, batch_size, device, single_label):
    saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, batch_size, device)
    saved_model_evaluator.single_label = single_label
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
        'SST-1': SST1,
        'SST-2': SST2,
        'Reuters': Reuters,
        'AAPD': AAPD,
        'IMDB': IMDB,
        'Yelp2014':Yelp2014
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
        model = XmlCNN(config)
        if args.cuda:
            model.cuda()
            print('Shift model to GPU')

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    #optimizer = torch.optim.Adadelta(parameter, lr=args.lr, weight_decay=args.weight_decay)
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
    trainer_config = {
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'log_interval': args.log_every,
        'dev_log_interval': args.dev_every,
        'patience': args.patience,
        'model_outfile': args.save_path,   # actually a directory, using model_outfile to conform to Trainer naming convention
        'logger': logger,
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

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')
    else:
        evaluate_dataset('dev', dataset_map[args.dataset], model, None, dev_iter, args.batch_size, args.gpu, args.single_label)
        evaluate_dataset('test', dataset_map[args.dataset], model, None, test_iter, args.batch_size, args.gpu, args.single_label)

    if args.onnx:
        device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')
        dummy_input = torch.zeros(args.onnx_batch_size, args.onnx_sent_len, dtype=torch.long, device=device)
        onnx_filename = 'xmlcnn_{}.onnx'.format(args.mode)
        torch.onnx.export(model, dummy_input, onnx_filename)
        print('Exported model in ONNX format as {}'.format(onnx_filename))
