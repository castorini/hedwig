"""import logging
import random
from copy import deepcopy

import numpy as np
import torch

from common.evaluate import EvaluatorFactory
from common.train import TrainerFactory
from datasets.aapd import AAPD
from datasets.imdb import IMDB
from datasets.reuters import Reuters
from datasets.yelp2014 import Yelp2014
from models.bert.args import get_args
from models.bert.model import BertForSequenceClassification


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

    scores, metric_names = saved_model_evaluator.get_scores()
    print('Evaluation metrics for', split_name)
    print(metric_names)
    print(scores)


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
        'Reuters': Reuters,
        'AAPD': AAPD,
        'IMDB': IMDB,
        'Yelp2014': Yelp2014
    }

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    else:
        dataset_class = dataset_map[args.dataset]
        train_iter, dev_iter, test_iter = dataset_class.iters(args.data_dir,
                                                              args.word_vectors_file,
                                                              args.word_vectors_dir,
                                                              batch_size=args.batch_size,
                                                              device=args.gpu,
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
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
    else:
        model = RegLSTM(config)
        if args.cuda:
            model.cuda()

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)

    train_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, train_iter, args.batch_size, args.gpu)
    test_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, test_iter, args.batch_size, args.gpu)
    dev_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, dev_iter, args.batch_size, args.gpu)

    if hasattr(train_evaluator, 'is_multilabel'):
        train_evaluator.is_multilabel = dataset_class.IS_MULTILABEL
    if hasattr(test_evaluator, 'is_multilabel'):
        test_evaluator.is_multilabel = dataset_class.IS_MULTILABEL
    if hasattr(dev_evaluator, 'is_multilabel'):
        dev_evaluator.is_multilabel = dataset_class.IS_MULTILABEL

    trainer_config = {
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'log_interval': args.log_every,
        'patience': args.patience,
        'model_outfile': args.save_path,
        'logger': logger,
        'is_multilabel': dataset_class.IS_MULTILABEL
    }

    trainer = TrainerFactory.get_trainer(args.dataset, model, None, train_iter, trainer_config, train_evaluator, test_evaluator, dev_evaluator)

    if not args.trained_model:
        trainer.train(args.epochs)
    else:
        if args.cuda:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage)

    model = torch.load(trainer.snapshot_path)

    if model.beta_ema > 0:
        old_params = model.get_params()
        model.load_ema_params()

    # Calculate dev and test metrics
    evaluate_dataset('dev', dataset_class, model, None, dev_iter, args.batch_size,
                     is_multilabel=dataset_class.IS_MULTILABEL,
                     device=args.gpu)
    evaluate_dataset('test', dataset_class, model, None, test_iter, args.batch_size,
                     is_multilabel=dataset_class.IS_MULTILABEL,
                     device=args.gpu)

    if model.beta_ema > 0:
        model.load_params(old_params)"""
from models.bert.args import get_args
from models.bert.model import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from __future__ import absolute_import, division, print_function
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from datasets.dataset4bert import DataProcessor, Sst2Processor, ReutersProcessor, AAPDProcessor, InputExample, InputFeatures
from datasets.tokenization4bert import BertTokenizer
from common.train4bert import TrainerFactory
from common.evaluate4bert import EvaluatorFactory


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()
    #logger = get_logger()

    # Set random seed for reproducibility

    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
    }

    num_labels_task = {
        "cola": 2,
        "sst-2": 2,
        "mnli": 3,
        "mrpc": 2,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None


    ########################
    ## Trainer Config, Eval config, Trainer, Evaluator : Just need to modularize trainer and evaluator similar to #hedwig
    #trainer_config = {
}
    #trainer = TrainerFactory.get_trainer(args.)

    #evaluator = EvaluatorFactory.get_evaluator('...')
    ########################
    if args.do_train:
        trainer.train()
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0): # Dont need the second condition here perhaps
        

