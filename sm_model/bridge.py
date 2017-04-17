import os
import sys
import pickle
import string
from collections import defaultdict

import numpy as np
import torch
from nltk.tokenize import TreebankWordTokenizer
from torch.autograd import Variable

from sm_model import model

sys.modules['model'] = model


class SMModelBridge(object):

    def __init__(self, model_file, word_embeddings_cache_file, stopwords_file, word2dfs_file):
        # init torch random seeds
        torch.manual_seed(1234)
        np.random.seed(1234)

        # load model
        self.model = model.QAModel.load(model_file)
        # load vectors
        self.vec_dim = self._preload_cached_embeddings(word_embeddings_cache_file)
        self.unk_term_vec = np.random.uniform(-0.25, 0.25, self.vec_dim)

        # stopwords
        self.stoplist = set([line.strip() for line in open(stopwords_file)])

        # word dfs
        if os.path.isfile(word2dfs_file):
            with open(word2dfs_file, "rb") as w2dfin:
                self.word2dfs = pickle.load(w2dfin)


    def _preload_cached_embeddings(self, cache_file):

        with open(cache_file + '.dimensions') as d:
            vocab_size, vec_dim = [int(e) for e in d.read().strip().split()]

        self.W = np.memmap(cache_file, dtype=np.double, shape=(vocab_size, vec_dim))

        with open(cache_file + '.vocab') as f:
            w2v_vocab_list = map(str.strip, f.readlines())

        self.vocab_dict = {w:k for k, w in enumerate(w2v_vocab_list)}
        return vec_dim


    def parser(self, q, a):
        q_toks = TreebankWordTokenizer().tokenize(q)
        q_str = ' '.join(q_toks).lower()
        a_list = []
        for ans in a:
            ans_toks = TreebankWordTokenizer().tokenize(ans)
            a_str = ' '.join(ans_toks).lower()
            a_list.append(a_str)
        return q_str, a_list


    def compute_overlap_features(self, q_str, a_list, word2df=None, stoplist=None):
        word2df = word2df if word2df else {}
        stoplist = stoplist if stoplist else set()
        feats_overlap = []
        for a in a_list:
            question = q_str.split()
            answer = a.split()
            # q_set = set(question)
            # a_set = set(answer)
            q_set = set([q for q in question if q not in stoplist])
            a_set = set([a for a in answer if a not in stoplist])
            word_overlap = q_set.intersection(a_set)
            # overlap = float(len(word_overlap)) / (len(q_set) * len(a_set) + 1e-8)
            if len(q_set) == 0 and len(a_set) == 0:
                overlap = 0
            else:
                overlap = float(len(word_overlap)) / (len(q_set) + len(a_set))

            # q_set = set([q for q in question if q not in stoplist])
            # a_set = set([a for a in answer if a not in stoplist])
            word_overlap = q_set.intersection(a_set)
            df_overlap = 0.0
            for w in word_overlap:
                df_overlap += word2df[w]

            if len(q_set) == 0 and len(a_set) == 0:
                df_overlap = 0
            else:
                df_overlap /= (len(q_set) + len(a_set))

            feats_overlap.append(np.array([overlap, df_overlap]))
        return np.array(feats_overlap)


    def make_input_matrix(self, sentence):
        terms = sentence.strip().split()
        # word_embeddings = torch.zeros(max_len, vec_dim).type(torch.DoubleTensor)
        word_embeddings = torch.zeros(len(terms), self.vec_dim).type(torch.DoubleTensor)
        for i in range(len(terms)):
            word = terms[i]
            if word not in self.vocab_dict:
                emb = torch.from_numpy(self.unk_term_vec)
            else:
                emb = torch.from_numpy(self.W[self.vocab_dict[word]])
            word_embeddings[i] = emb
        input_tensor = torch.zeros(1, self.vec_dim, len(terms))
        input_tensor[0] = torch.transpose(word_embeddings, 0, 1)
        return input_tensor


    def get_tensorized_inputs(self, batch_ques, batch_sents, batch_ext_feats):
        assert(1 == len(batch_ques))
        tensorized_inputs = []
        for i in range(len(batch_ques)):
            xq = Variable(self.make_input_matrix(batch_ques[i]))
            xs = Variable(self.make_input_matrix(batch_sents[i]))
            ext_feats = Variable(torch.FloatTensor(batch_ext_feats[i]))
            ext_feats = torch.unsqueeze(ext_feats, 0)
            tensorized_inputs.append((xq, xs, ext_feats))
        return tensorized_inputs


    def rerank_candidate_answers(self, question, answers):
        # tokenize
        q_str, a_list = self.parser(question, answers)

        # calculate overlap features
        overlap_feats = self.compute_overlap_features(q_str, a_list, \
            stoplist=None, word2df=self.word2dfs)
        overlap_feats_stoplist = self.compute_overlap_features(q_str, a_list, \
            stoplist=self.stoplist, word2df=self.word2dfs)
        overlap_feats_vec = np.hstack([overlap_feats, overlap_feats_stoplist])

        # run through the model
        scores_sentences = []
        for i in range(len(a_list)):
            xq, xa, x_ext_feats = self.get_tensorized_inputs([q_str], [a_list[i]], \
                [overlap_feats_vec[i]])[0]
            pred = self.model(xq, xa, x_ext_feats)
            pred = torch.exp(pred)
            scores_sentences.append((pred.data.squeeze()[1], a_list[i]))

        return scores_sentences


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Bridge Demo. Produces scores in trec_eval format")
    ap.add_argument('model')
    ap.add_argument('--word_embeddings_cache', default='../data/word2vec/aquaint+wiki.txt.gz.ndim=50.cache')
    ap.add_argument('--stopwords_file', default='../data/TrecQA/stopwords.txt')
    ap.add_argument('--wordDF_file', default='../data/TrecQA/word2dfs.p')
    ap.add_argument('--no_ext_feats', action="store_true", help="This argument has no effect because the model saves its members")
    ap.add_argument('--use_pre_ext_feats', action="store_true", help="use the precomputed external overlap features")
    ap.add_argument('--data_folder', default='../data/TrecQA/')
    ap.add_argument('dataset', choices=['train-all', 'raw-test', 'raw-dev', 'train'])
    ap.add_argument('out_scorefile', help='file in trec_eval format')
    ap.add_argument('--out_qrels', help='will also output qrels trec_eval format')

    args = ap.parse_args()

    smmodel = SMModelBridge(
            #'../models/sm_model/sm_model.TrecQA.TRAIN-ALL.2017-04-02.castor',
            args.model,
            args.word_embeddings_cache,
            args.stopwords_file,
            args.wordDF_file)
    
    # if args.no_ext_feats:
    #     smmodel.model.no_ext_feats = True


    allque = [q.strip() for q in open(os.path.join('../data/TrecQA/', args.dataset+'/a.toks')).readlines()]
    allans = [a.strip() for a in open(os.path.join('../data/TrecQA/', args.dataset+'/b.toks')).readlines()]
    labels = [y.strip() for y in open(os.path.join('../data/TrecQA/', args.dataset+'/sim.txt')).readlines()]
    qids = [id.strip() for id in open(os.path.join('../data/TrecQA/', args.dataset+'/id.txt')).readlines()]

    pre_ext_feats = None
    if args.use_pre_ext_feats:
        pre_ext_feats = [ [float(e) for e in x.split() ] for x in open(os.path.join('../data/TrecQA/', args.dataset+'/overlap_feats.txt')).readlines()]
    
    scoref = open(args.out_scorefile, 'w')
    if args.out_qrels:
        qrelf = open(args.out_qrels, 'w')

    for i in range(len(allque)):
        question = allque[i]
        answers = [allans[i]]
        ext_feats = None
        if args.use_pre_ext_feats:
            ext_feats = [pre_ext_feats[i]]
        ss = smmodel.rerank_candidate_answers(question, answers, ext_feats)
        # print('Question:', question)
        for score, sentence in ss:
            #print(score, '\t', sentence)
            #print('{}\t{}'.format(labels[i], score))
            print('{} {} {} {} {} {}'.format(qids[i], '0', i, 0, score, 'sm_model.'+args.dataset), file=scoref)
            if args.out_qrels:
                print('{} {} {} {}'.format(qids[i], '0', i, labels[i]), file=qrelf)
