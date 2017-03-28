import os 
import sys
import time
import glob
import argparse
import numpy as np

import pandas as pd
import subprocess
import shlex

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable


from model import QAModel
import utils
from train import Trainer

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def logargs(func):
    def inner(*args, **kwargs):
        logger.info('%s : %s %s' % (func.__name__, args, kwargs))
        return func(*args, **kwargs)
    return inner


def compute_map_mrr(dataset_folder, set_folder, test_scores):
    logger.info( "Running trec_eval script..." )
    N = len(test_scores)

    qids_test, y_test = utils.get_test_qids_labels(dataset_folder, set_folder)

    # Call TrecEval code to calc MAP and MRR
    df_submission = pd.DataFrame(index=np.arange(N), columns=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id'])
    df_submission['qid'] = qids_test
    df_submission['iter'] = 0
    df_submission['docno'] = np.arange(N)
    df_submission['rank'] = 0
    df_submission['sim'] = test_scores
    df_submission['run_id'] = 'smmodel'
    df_submission.to_csv(os.path.join(args.dataset_folder, 'submission.txt'), header=False, index=False, sep=' ')

    df_gold = pd.DataFrame(index=np.arange(N), columns=['qid', 'iter', 'docno', 'rel'])
    df_gold['qid'] = qids_test
    df_gold['iter'] = 0
    df_gold['docno'] = np.arange(N)
    df_gold['rel'] = y_test
    df_gold.to_csv(os.path.join(args.dataset_folder, 'gold.txt'), header=False, index=False, sep=' ')

    # subprocess.call("/bin/sh run_eval.sh '{}'".format(args.dataset_folder), shell=True)
    pargs = shlex.split("/bin/sh run_eval.sh '{}'".format(args.dataset_folder))
    p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pout, perr = p.communicate()    
    lines = pout.split('\n')
    map = float(lines[0].strip().split()[-1])
    mrr = float(lines[1].strip().split()[-1])
    return map, mrr



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='pytorch port of the SM model')
    ap.add_argument('word_vectors_file', help='NOTE: a cache will be created for faster loading for word vectors')
    ap.add_argument('dataset_folder', help='directory containing train, dev, test sets')
    ap.add_argument('model_fname', help='model will be saved in args.dataset_folder/<model_fname>')
    ap.add_argument('--classes', type=int, default=2)

    # system arguments
    # TODO: add arguments for CUDA
    ap.add_argument('--num_threads', help="the number of simultaneous processes to run", type=int, default=4)

    # training arguments
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--filter_width', type=int, default=5)    
    ap.add_argument('--eta', help='Initial learning rate', default=0.001, type=float)
    ap.add_argument('--mom', help='SGD Momentum', default=0.0, type=float)    
    
    # epoch related arguments
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--patience', type=int, default=5, help="if there is no appreciable change in model after <patience> epochs, then stop")
    
    # debugging arguments
    ap.add_argument('--debugSingleBatch', action="store_true", help="will stop program after training 1 input batch")
    ap.add_argument('--num_conv_filters', help="the number of convolution channels (lesser is faster)", default=100, type=int)
    ap.add_argument('--no_ext_feats', action="store_true", help="will not include external features in the model")    
    ap.add_argument('--no_loss_reg', help="no loss regularization", action="store_true")

    args = ap.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    # cache word embeddings
    cache_file = os.path.splitext(args.word_vectors_file)[0] + '.cache'    
    utils.cache_word_embeddings(args.word_vectors_file, cache_file)

    vocab_size, vec_dim = utils.load_embedding_dimensions(cache_file)
    
    # instantiate model
    net = QAModel(vec_dim, args.filter_width, args.num_conv_filters, args.no_ext_feats) #filter width is 5
    QAModel.save(net, args.dataset_folder, args.model_fname)

    torch.set_num_threads(args.num_threads)
    
    trainer = Trainer(net, args.eta, args.mom, args.no_loss_reg)
    trainer.load_input_data(args.dataset_folder, cache_file, 'train', 'clean-dev', 'clean-test')

    best_map = 0.0
    best_model = 0
    
    for i in range(args.epochs):
        logger.info('------------- Training epoch {} --------------'.format(i+1))        
        train_accuracy = trainer.train('train', args.batch_size, args.debugSingleBatch)        
        if args.debugSingleBatch: sys.exit(0)
        dev_scores = trainer.test('clean-dev', args.batch_size)

        dev_map, dev_mrr = compute_map_mrr(args.dataset_folder, 'clean-dev', dev_scores)
        logger.info("------- MAP {}, MRR {}".format(dev_map, dev_mrr))

        if np.fabs(dev_map - best_map) > 1e-3:            
            best_model = i
            best_map = dev_map
            QAModel.save(net, args.dataset_folder, args.model_fname)
            logger.info('Achieved better dev_map ... saved model')
        
        if (i - best_model) >= args.patience:
            logger.warning('No improvement since the last {} epochs. Stopping training'.format(i - best_model))
            break
    
    logger.info(' ------------ Training epochs completed!')        
    logger.info('Best MAP in training phase = {:.4f}'.format(best_map))

    model = QAModel.load(args.dataset_folder, args.model_fname)
    evaluator = Trainer(model, args.eta, args.mom, args.no_loss_reg)
    evaluator.load_input_data(args.dataset_folder, cache_file, None, None, 'clean-test')
    test_scores = evaluator.test('clean-test', args.batch_size)
    
    map, mrr = compute_map_mrr(args.dataset_folder, 'clean-test', test_scores)
    logger.info("------- MAP {}, MRR {}".format(map, mrr))

    
