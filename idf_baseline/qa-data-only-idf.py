import argparse
import os
import numpy as np
from collections import defaultdict
import string

import nltk
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def read_in_data(datapath, set_name, file, stop_and_stem=False):
    data = []
    with open(os.path.join(datapath, set_name, file)) as inf:
        data = [line.strip() for line in inf.readlines()]
        if stop_and_stem:
            stemmer = PorterStemmer()
            stoplist = set(stopwords.words('english'))
            stoplist.update(set(string.punctuation))
            def stop_stem(sentence):
                return ' '.join([stemmer.stem(word) for word in sentence.split() \
                                                        if word not in stoplist])
            data = [stop_stem(sentence) for sentence in data]
    return data


def compute_idfs(data):
    term_idfs = defaultdict(float)
    for doc in list(data):
        for term in list(set(doc.split())):
            term_idfs[term] += 1.0
    N = len(data)
    for term, n_t in term_idfs.items():
        term_idfs[term] = np.log(N/(1+n_t))
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
            print('{} 0 {} 0 {} data_only_idfbaseline'.format(qids[i], docid_c,
                                                              idf_sum_similarity[i]),
                  file=outf)
            docid_c += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="uses idf weights from the question-answer pairs only,\
                   and not from the whole corpus")
    ap.add_argument('qa_data', help="path to the QA dataset",
                    choices=['../../data/TrecQA', '../../data/WikiQA'])
    ap.add_argument('outfile_prefix', help="output file prefix")
    ap.add_argument('--ignore-test', help="does not consider test data when computing IDF of terms",
                    action="store_true")
    ap.add_argument("--stop-and-stem", help='performs stopping and stemming', action="store_true")

    args = ap.parse_args()

    # read in the data
    train_data, dev_data, test_data = 'train', 'dev', 'test'
    if args.qa_data.endswith('TrecQA'):
        train_data, dev_data, test_data = 'train-all', 'raw-dev', 'raw-test'

    train_que = read_in_data(args.qa_data, train_data, 'a.toks', args.stop_and_stem)
    train_ans = read_in_data(args.qa_data, train_data, 'b.toks', args.stop_and_stem)

    dev_que = read_in_data(args.qa_data, dev_data, 'a.toks', args.stop_and_stem)
    dev_ans = read_in_data(args.qa_data, dev_data, 'b.toks', args.stop_and_stem)

    test_que = read_in_data(args.qa_data, test_data, 'a.toks', args.stop_and_stem)
    test_ans = read_in_data(args.qa_data, test_data, 'b.toks', args.stop_and_stem)

    all_data = train_que + dev_que + train_ans + dev_ans

    if not args.ignore_test:
        all_data += test_ans
        all_data += test_que

    # compute inverse document frequencies for terms
    term_idfs = compute_idfs(set(all_data))

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

