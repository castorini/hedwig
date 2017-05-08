import json
import os
import sys
from collections import Counter
import argparse

import re
import string
import numpy as np
import torch
from nltk.tokenize import TreebankWordTokenizer
from torch.autograd import Variable
from py4j.java_gateway import JavaGateway

from sm_cnn import model
from sm_cnn.external_features import compute_overlap, compute_idf_weighted_overlap, stopped

sys.modules['model'] = model


class SMModelBridge(object):

    def __init__(self, model_file, word_embeddings_cache_file, index_path):
        # init torch random seeds
        torch.manual_seed(1234)
        np.random.seed(1234)

        # load model
        self.model = model.QAModel.load(model_file)
        self.model_file = model_file

        # load vectors
        self.vec_dim = self._preload_cached_embeddings(word_embeddings_cache_file)
        self.unk_term_vec = np.random.uniform(-0.25, 0.25, self.vec_dim)

        self.index = index_path


    def _preload_cached_embeddings(self, cache_file):

        with open(cache_file + '.dimensions') as d:
            vocab_size, vec_dim = [int(e) for e in d.read().strip().split()]

        self.W = np.memmap(cache_file, dtype=np.double, shape=(vocab_size, vec_dim))

        with open(cache_file + '.vocab') as f:
            w2v_vocab_list = map(str.strip, f.readlines())

        self.vocab_dict = {w:k for k, w in enumerate(w2v_vocab_list)}
        return vec_dim


    def parse(self, sentence):
        s_toks = TreebankWordTokenizer().tokenize(sentence)
        sentence = ' '.join(s_toks).lower()

        model_input_args = self.model_file.split('.')
        punctuation = model_input_args[-3].split('-')[1]
        dash_words = model_input_args[-2].split('_')[1]

        if dash_words == "split":
            def split_hyphenated_words(sentence):
                rtokens = []
                for term in sentence.split():
                    for t in term.split('-'):
                        if t:
                            rtokens.append(t)
                return ' '.join(rtokens)
            sentence = split_hyphenated_words(sentence)

        if punctuation == "remove":
            regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
            def remove_punctuation(sentence):
                rtokens = []
                for term in sentence.split():
                    for t in regex.sub(' ', term).strip().split():
                        if t:
                            rtokens.append(t)
                return ' '.join(rtokens)
            sentence = remove_punctuation(sentence)

        return sentence


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

    def rerank_candidate_answers(self, question, answers, idf_json):
        # run through the model
        scores_sentences = []
        question = self.parse(question)
        term_idfs = json.loads(idf_json)
        term_idfs = dict((k, float(v)) for k, v in term_idfs.items())

        for term in question.split():
            if term not in term_idfs:
                term_idfs[term] = 0.0

        for answer in answers:
            answer = self.parse(answer)
            for term in answer.split():
                if term not in term_idfs:
                    term_idfs[term] = 0.0
    
            overlap = compute_overlap([question], [answer])
            idf_weighted_overlap = compute_idf_weighted_overlap([question], [answer], term_idfs)
            overlap_no_stopwords =\
                compute_overlap(stopped([question]), stopped([answer]))
            idf_weighted_overlap_no_stopwords =\
                compute_idf_weighted_overlap(stopped([question]), stopped([answer]), term_idfs)
            ext_feats = [np.array(feats) for feats in zip(overlap, idf_weighted_overlap,\
                        overlap_no_stopwords, idf_weighted_overlap_no_stopwords)]

            xq, xa, x_ext_feats = self.get_tensorized_inputs([question], [answer], \
                ext_feats)[0]
            pred = self.model(xq, xa, x_ext_feats)
            pred = torch.exp(pred)
            scores_sentences.append((pred.data.squeeze()[1], answer))

        return scores_sentences

def get_term_idf_json_list(index_path, sent_list):
    gateway = JavaGateway()
    index = gateway.jvm.java.lang.String(index_path)
    pyserini = gateway.jvm.io.anserini.py4j.PyseriniEntryPoint()
    pyserini.initializeWithIndex(index_path)
    java_list = gateway.jvm.java.util.ArrayList()

    for l in sent_list:
        java_list.add(l)

    json_object = pyserini.getTermIdfJSONs(java_list)
    return json_object

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bridge Demo. Produces scores in trec_eval format",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('model', help="the path to the saved model file")
    ap.add_argument('--word-embeddings-cache', help="the embeddings 'cache' file",\
        default='../data/word2vec/aquaint+wiki.txt.gz.ndim=50.cache')
    ap.add_argument('index_path', help="the path to the source corpus index")
    # ap.add_argument('--paper-ext-feats', action="store_true", \
    #     help="external features as per the paper")
    ap.add_argument('--dataset-folder', help="the QA dataset folder {TrecQA|WikiQA}",
                    default='../data/TrecQA/')

    args = ap.parse_args()

    smmodel = SMModelBridge(
        args.model,
        args.word_embeddings_cache,
        args.index_path
        )

    train_set, dev_set, test_set = 'train', 'dev', 'test'
    if 'TrecQA' in args.dataset_folder:
        train_set, dev_set, test_set = 'train-all', 'raw-dev', 'raw-test'



    for split in [dev_set, test_set]:
        outfile = open('bridge.{}.scores'.format(split), 'w')

        questions = [q.strip() for q in \
                        open(os.path.join(args.dataset_folder, split, 'a.toks')).readlines()]
        answers = [q.strip() for q in \
                        open(os.path.join(args.dataset_folder, split, 'b.toks')).readlines()]
        labels = [q.strip() for q in \
                        open(os.path.join(args.dataset_folder, split, 'sim.txt')).readlines()]
        qids = [q.strip() for q in \
                        open(os.path.join(args.dataset_folder, split, 'id.txt')).readlines()]

        qid_question = dict(zip(qids, questions))
        q_counts = Counter(questions)

        answers_offset = 0
        docid_counter = 0

        all_questions_answers = questions + answers
        idf_json = get_term_idf_json_list(args.index_path, all_questions_answers)

        for qid, question in sorted(qid_question.items(), key=lambda x: float(x[0])):
            num_answers = q_counts[question]
            q_answers = answers[answers_offset: answers_offset + num_answers]
            answers_offset += num_answers
            sentence_scores = smmodel.rerank_candidate_answers(question, q_answers, idf_json)

            for score, sentence in sentence_scores:
                print('{} Q0 {} 0 {} sm_cnn_bridge.{}.run'.format(
                    qid,
                    docid_counter,
                    score,
                    os.path.basename(args.dataset_folder)
                ), file=outfile)
                docid_counter += 1
            if 'WikiQA' in args.dataset_folder:
                docid_counter = 0

        outfile.close()
