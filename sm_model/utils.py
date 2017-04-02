# file input output

import os
import sys

import numpy as np
import torch
from gensim.models.keyedvectors import KeyedVectors

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


def cache_word_embeddings(word_embeddings_file, cache_file):
    if not word_embeddings_file.endswith('.gz'):
        logger.warning('WARNING: expecting a .gz file. Is the {} in the correct \
            format?'.format(word_embeddings_file))

    vocab_size, vec_dim = 0, 0

    if not os.path.exists(cache_file):
        # cache does not exist
        if not os.path.exists(os.path.dirname(cache_file)):
            # make cache folder if needed
            os.mkdir(os.path.dirname(cache_file))
        logger.info('caching the word embeddings in np.memmap format')
        wv = KeyedVectors.load_word2vec_format(word_embeddings_file, binary=True)
        # print len(wv.syn0), wv.syn0.shape
        # print len(wv.syn0norm) if wv.syn0norm else None
        fp = np.memmap(cache_file, dtype=np.double, mode='w+', shape=wv.syn0.shape)
        fp[:] = wv.syn0[:]
        with open(cache_file + '.vocab', 'w') as f:
            logger.info('writing out vocab for {}'.format(word_embeddings_file))
            for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
                print(w.encode('utf-8'), file=f)
        with open(cache_file + '.dimensions', 'w') as f:
            logger.info('writing out dimensions for {}'.format(word_embeddings_file))
            print(wv.syn0.shape[0], wv.syn0.shape[1], file=f)
        vocab_size, vec_dim = wv.syn0.shape
        del fp, wv
        print('cached {} into {}'.format(word_embeddings_file, cache_file))

    return vocab_size, vec_dim


def load_embedding_dimensions(cache_file):
    vocab_size, vec_dim = 0, 0
    with open(cache_file + '.dimensions') as d:
        vocab_size, vec_dim = [int(e) for e in d.read().strip().split()]
    return vocab_size, vec_dim

def load_cached_embeddings(cache_file, vocab_list, oov_vec=[]):
    logger.debug('loading cached embeddings ')

    with open(cache_file + '.dimensions') as d:
        vocab_size, vec_dim = [int(e) for e in d.read().strip().split()]

    W = np.memmap(cache_file, dtype=np.double, shape=(vocab_size, vec_dim))

    with open(cache_file + '.vocab') as f:
        logger.debug('loading vocab')
        w2v_vocab_list = map(str.strip, f.readlines())

    vocab_dict = {w:k for k, w in enumerate(w2v_vocab_list)}

    # Read w2v for vocab appears in Q and A
    w2v_dict = {}
    for word in vocab_list:
        if word in w2v_dict:
            continue
        if word in vocab_dict:
            w2v_dict[word] = W[vocab_dict[word]]
        else:
            w2v_dict[word] = np.random.uniform(-0.25, 0.25, vec_dim) \
                if len(oov_vec) == 0 else oov_vec
            #w2v_dict[word] = W[vocab_dict["unk"]]
    return w2v_dict


def read_in_dataset(dataset_folder, set_folder):
    """
    read in the data to return (question, sentence, label)
    set_folder = {train|dev|test}
    """
    max_q = 0
    max_s = 0
    set_path = os.path.join(dataset_folder, set_folder)

    questions = [line.strip() for line in open(os.path.join(set_path, 'a.toks')).readlines()]
    len_q_list = [len(q.split()) for q in questions]

    sentences = [line.strip() for line in open(os.path.join(set_path, 'b.toks')).readlines()]
    len_s_list = [len(s.split()) for s in sentences]

    labels = [int(line.strip()) for line in open(os.path.join(set_path, 'sim.txt')).readlines()]
    ext_feats = np.array([list(map(float, line.strip().split(' '))) \
        for line in open(os.path.join(set_path, 'overlap_feats.txt')).readlines()])

    #y = torch.from_numpy(labels)
    #return questions, sentences, y

    vocab = [line.strip() for line in open(os.path.join(dataset_folder, 'vocab.txt')).readlines()]
    return questions, sentences, labels, vocab, max(len_q_list), max(len_s_list), ext_feats


def get_test_qids_labels(dataset_folder, set_folder):
    set_path = os.path.join(dataset_folder, set_folder)
    qids = [line.strip() for line in open(os.path.join(set_path, 'id.txt')).readlines()]
    labels = np.array([int(line.strip()) for line in open(os.path.join(set_path, 'sim.txt')).readlines()])
    return qids, labels


if __name__ == "__main__":

    vocab = ["unk", "idontreallythinkthiswordexists", "hello"]

    w2v_dict, vec_dim = load_cached_embeddings("../../data/word2vec/aquaint+wiki.txt.gz.ndim=50.cache", vocab)

    for w, v in w2v_dict.iteritems():
        print(w)
        print(v)
