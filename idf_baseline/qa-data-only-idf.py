import argparse
import os
import sys
import re
import numpy as np
from collections import defaultdict
import string
import subprocess
import shlex

import nltk
nltk.download('stopwords', quiet=True)

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def read_in_data(datapath, set_name, file, stop_and_stem=False, stop_punct=False, dash_split=False):
    data = []
    with open(os.path.join(datapath, set_name, file)) as inf:
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


def compute_idfs(data, dash_split=False):    
    term_idfs = defaultdict(float)
    for doc in list(data):
        for term in list(set(doc.split())):
            if dash_split:
                assert('-' not in term)
            term_idfs[term] += 1.0
    N = len(data)
    for term, n_t in term_idfs.items():
        term_idfs[term] = np.log(N/(1+n_t))
    return term_idfs

def fetch_idfs_from_index(data, dash_split, indexPath):
    regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
    term_idfs = defaultdict(float)
    all_terms = set([term for doc in list(data) for term in doc.split()])
    with open('dataset.vocab', 'w') as vf:
        for term in list(all_terms):
            if dash_split:
                assert('-' not in term)
            print(term, file=vf)

    fetchIDF_cmd = \
        "sh ../idf_baseline/target/appassembler/bin/FetchTermIDF -index {} -vocabFile {}".\
            format(indexPath, 'dataset.vocab')
    pargs = shlex.split(fetchIDF_cmd)
    p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, \
                             bufsize=1, universal_newlines=True)
    pout, perr = p.communicate()

    lines = str(pout).split('\n')
    for line in lines:
        if not line:
            continue
        fields = line.strip().split("\t")
        term, weight = fields[0], fields[-1]
        term_idfs[term] = float(weight)

    for line in str(perr).split('\n'):
        print('Warning: '+line)
    return term_idfs

def compute_idf_sum_similarity(questions, answers, term_idfs):
    # compute IDF sums for common_terms
    idf_sum_similarity = np.zeros(len(questions))
    for i in range(len(questions)):
        q = questions[i]
        a = answers[i]
        q_terms = set(q.split())
        a_terms = set(a.split())
        common_terms = q_terms.intersection(a_terms)
        idf_sum_similarity[i] = np.sum([term_idfs[term] for term in list(common_terms)])

    return idf_sum_similarity


def write_out_idf_sum_similarities(qids, questions, answers, term_idfs, outfile, dataset):
    with open(outfile, 'w') as outf:
        idf_sum_similarity = compute_idf_sum_similarity(questions, answers, term_idfs)
        old_qid = 0
        docid_c = 0
        for i in range(len(questions)):
            if qids[i] != old_qid and dataset.endswith('WikiQA'):
                docid_c = 0
                old_qid = qids[i]
            print('{} 0 {} 0 {} idfbaseline'.format(qids[i], docid_c,
                                                              idf_sum_similarity[i]),
                  file=outf)
            docid_c += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="uses idf weights from the question-answer pairs only,\
                   and not from the whole corpus")
    ap.add_argument('qa_data', help="path to the QA dataset",
                    choices=['../../Castor-data/TrecQA', '../../Castor-data/WikiQA'])
    ap.add_argument('outfile_prefix', help="output file prefix")
    ap.add_argument('--ignore-test', help="does not consider test data when computing IDF of terms",
                    action="store_true")
    ap.add_argument("--stop-and-stem", help='performs stopping and stemming', action="store_true")
    ap.add_argument("--stop-punct", help='removes punctuation', action="store_true")
    ap.add_argument("--dash-split", help="split words containing hyphens", action="store_true")
    ap.add_argument("--index-for-corpusIDF", help="fetches idf from Index. provide index path. will\
    generate a vocabFile")

    args = ap.parse_args()

    # read in the data
    train_data, dev_data, test_data = 'train', 'dev', 'test'
    if args.qa_data.endswith('TrecQA'):
        train_data, dev_data, test_data = 'train-all', 'raw-dev', 'raw-test'

    train_que = read_in_data(args.qa_data, train_data, 'a.toks',
                             args.stop_and_stem, args.stop_punct, args.dash_split)
    train_ans = read_in_data(args.qa_data, train_data, 'b.toks',
                             args.stop_and_stem, args.stop_punct, args.dash_split)

    dev_que = read_in_data(args.qa_data, dev_data, 'a.toks',
                           args.stop_and_stem, args.stop_punct, args.dash_split)
    dev_ans = read_in_data(args.qa_data, dev_data, 'b.toks',
                           args.stop_and_stem, args.stop_punct, args.dash_split)

    test_que = read_in_data(args.qa_data, test_data, 'a.toks',
                            args.stop_and_stem, args.stop_punct, args.dash_split)
    test_ans = read_in_data(args.qa_data, test_data, 'b.toks',
                            args.stop_and_stem, args.stop_punct, args.dash_split)

    all_data = train_que + dev_que + train_ans + dev_ans

    if not args.ignore_test:
        all_data += test_ans
        all_data += test_que

    # compute inverse document frequencies for terms
    if not args.index_for_corpusIDF:
        term_idfs = compute_idfs(set(all_data), args.dash_split)
    else:
        term_idfs = fetch_idfs_from_index(set(all_data), args.dash_split, args.index_for_corpusIDF)

    # write out in trec_eval format
    write_out_idf_sum_similarities(read_in_data(args.qa_data, train_data, 'id.txt'),
                                   train_que, train_ans, term_idfs,
                                   '{}.{}.idfsim'.format(args.outfile_prefix, train_data),
                                   args.qa_data)

    write_out_idf_sum_similarities(read_in_data(args.qa_data, dev_data, 'id.txt'),
                                   dev_que, dev_ans, term_idfs,
                                   '{}.{}.idfsim'.format(args.outfile_prefix, dev_data),
                                   args.qa_data)

    write_out_idf_sum_similarities(read_in_data(args.qa_data, test_data, 'id.txt'),
                                   test_que, test_ans, term_idfs,
                                   '{}.{}.idfsim'.format(args.outfile_prefix, test_data),
                                   args.qa_data)

