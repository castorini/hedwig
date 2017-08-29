# file input output
import os
import re
import string

from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

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
        with open(cache_file + '.vocab', 'w', encoding='utf-8') as f:
            logger.info('writing out vocab for {}'.format(word_embeddings_file))
            for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
                print(w, file=f)
        with open(cache_file + '.dimensions', 'w', encoding='utf-8') as f:
            logger.info('writing out dimensions for {}'.format(word_embeddings_file))
            print(wv.syn0.shape[0], wv.syn0.shape[1], file=f)
        vocab_size, vec_dim = wv.syn0.shape
        del fp, wv
        print('cached {} into {}'.format(word_embeddings_file, cache_file))

    return vocab_size, vec_dim


def load_embedding_dimensions(cache_file):
    vocab_size, vec_dim = 0, 0
    with open(cache_file + '.dimensions', encoding='utf-8') as d:
        vocab_size, vec_dim = [int(e) for e in d.read().strip().split()]
    return vocab_size, vec_dim


def load_cached_embeddings(cache_file, vocab_list, w2v_dict, oov_vec=[]):
    """
    w2v_dict is filled up as reference
    """
    logger.debug('loading cached embeddings ')

    with open(cache_file + '.dimensions', encoding='utf-8') as d:
        vocab_size, vec_dim = [int(e) for e in d.read().strip().split()]

    W = np.memmap(cache_file, dtype=np.double, shape=(vocab_size, vec_dim))

    with open(cache_file + '.vocab', encoding='utf-8') as f:
        logger.debug('loading vocab')
        w2v_vocab_list = map(str.strip, f.readlines())

    vocab_dict = {w:k for k, w in enumerate(w2v_vocab_list)}

    # Read w2v for vocab appears in Q and A
    for word in vocab_list:
        if word in w2v_dict:
            continue
        if word in vocab_dict:
            w2v_dict[word] = W[vocab_dict[word]]
        else:
            w2v_dict[word] = np.random.uniform(-0.25, 0.25, vec_dim) \
                if len(oov_vec) == 0 else oov_vec
            #w2v_dict[word] = W[vocab_dict["unk"]]

def read_in_data(datapath, set_name, file, stop_and_stem=False, stop_punct=False, dash_split=False):
    data = []
    with open(os.path.join(datapath, set_name, file), encoding='utf-8') as inf:
        data = [line.strip() for line in inf.readlines()]

        if dash_split:
            def split_hyphenated_words(sentence):
                rtokens = []
                for term in sentence.split():
                    for t in term.split('-'):
                        if t:
                            rtokens.append(t)
                return ' '.join(rtokens)
            data = [split_hyphenated_words(sentence) for sentence in data]

        if stop_punct:
            regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
            def remove_punctuation(sentence):
                rtokens = []
                for term in sentence.split():
                    for t in regex.sub(' ', term).strip().split():
                        if t:
                            rtokens.append(t)
                return ' '.join(rtokens)
            data = [remove_punctuation(sentence) for sentence in data]

        if stop_and_stem:
            stemmer = PorterStemmer()
            stoplist = set(stopwords.words('english'))
            def stop_stem(sentence):
                return ' '.join([stemmer.stem(word) for word in sentence.split() \
                                                        if word not in stoplist])
            data = [stop_stem(sentence) for sentence in data]
    return data

def read_in_dataset(dataset_folder, set_folder, stop_punct=False, dash_split=False):
    """
    read in the data to return (question, sentence, label)
    set_folder = {train|dev|test}
    """
    max_q = 0
    max_s = 0
    set_path = os.path.join(dataset_folder, set_folder)

    #questions = [line.strip() for line in open(os.path.join(set_path, 'a.toks')).readlines()]
    questions = read_in_data(dataset_folder, set_folder, "a.toks", False, stop_punct, dash_split)
    len_q_list = [len(q.split()) for q in questions]

    #sentences = [line.strip() for line in open(os.path.join(set_path, 'b.toks')).readlines()]
    sentences = read_in_data(dataset_folder, set_folder, "b.toks", False, stop_punct, dash_split)
    len_s_list = [len(s.split()) for s in sentences]

    #labels = [int(line.strip()) for line in open(os.path.join(set_path, 'sim.txt')).readlines()]
    labels = [int(lbl) for lbl in read_in_data(dataset_folder, set_folder, "sim.txt")]

    #vocab = [line.strip() for line in open(os.path.join(dataset_folder, 'vocab.txt')).readlines()]
    #all_data = list(set(questions)) + list(set(sentences))
    all_data = questions + sentences
    vocab_set = set()
    for sentence in all_data:
        for term in sentence.split():
            vocab_set.add(term)
    vocab = sorted(list(vocab_set))

    return [questions, sentences, labels, max(len_q_list), max(len_s_list), vocab]


def get_test_qids_labels(dataset_folder, set_folder):
    set_path = os.path.join(dataset_folder, set_folder)
    qids = [line.strip() for line in open(os.path.join(set_path, 'id.txt'), encoding='utf-8').readlines()]
    labels = np.array([int(line.strip()) for line in open(os.path.join(set_path, 'sim.txt'), encoding='utf-8').readlines()])
    return qids, labels


if __name__ == "__main__":

    vocab = ["unk", "idontreallythinkthiswordexists", "hello"]

    w2v_dict = {}
    load_cached_embeddings("../../data/word2vec/aquaint+wiki.txt.gz.ndim=50.cache", vocab, w2v_dict)

    for w, v in w2v_dict.iteritems():
        print(w)
        print(v)
