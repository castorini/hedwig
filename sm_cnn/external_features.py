# module to compute various external features for the sm cnn model.
# TODO: add more external features like:
# word mover distance, cosine sim in tf.idf space, cosine sim in word embedding space
# overlap based on parts of speech: noun, verb, adj (POS tag)
# word embedding cosine sim based on part of speech: noun, verb, adj
import sys
import os
import shlex
import subprocess
import string
from collections import defaultdict

import numpy as np

import nltk
nltk.download('stopwords', quiet=True)

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def stopped(sentences):
    """
    remove stop words from given sentences (questions|answers)
    """
    stoplist = set(stopwords.words('english'))
    #stoplist.update(set(string.punctuation))
    def stop(sentence):
        return ' '.join([word for word in sentence.split() if word not in stoplist])
    return [stop(sentence) for sentence in sentences]

def stemmed(sentences):
    """
    reduce sentence terms to stemmed representations
    """
    stemmer = PorterStemmer()
    def stem(sentence):
        return ' '.join([stemmer.stem(word) for word in sentence.split()])
    return [stem(sentence) for sentence in sentences]

def get_qadata_only_idf(all_data):
    """
    returns idf weights computed over all question answer pairs in the dataset
    """
    if not type(all_data) is list:
        all_data = list(all_data)
    term_idfs = defaultdict(float)
    for doc in all_data:
        for term in list(set(doc.split())):
            term_idfs[term] += 1.0
    N = len(all_data)
    for term, n_t in term_idfs.items():
        term_idfs[term] = np.log(N/(1+n_t))
    return term_idfs

def get_source_corpus_idf(all_data, path_to_index):
    """
    fetches idf weights from source corpus (disks1-5+aquaint|wikipedia) index, for all the qa pairs
    """
    # first run maven to build ../idf_baseline/FetchTermIDF
    maven_cmd = "mvn -f ../idf_baseline/pom.xml clean package appassembler:assemble"
    pargs = shlex.split(maven_cmd)
    p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, \
                             bufsize=1, universal_newlines=True)
    pout, perr = p.communicate()                             
    # if build failure, exit with message
    if "BUILD FAILURE" in pout or "BUILD FAILURE" in perr:
        print("\nERROR: Could not build ../idf_baseline/FetchTermIDF. Fix build errors before proceeding")
        print("$ cd ../idf_baseline")
        print("$ mvn clean package appassembler:assemble")
        sys.exit(0)

    if not type(all_data) is list:
        all_data = list(all_data)
    term_idfs = defaultdict(float)
    all_terms = set([term for doc in all_data for term in doc.split()])
    with open('dataset.vocab', 'w') as vf:
        for term in list(all_terms):
            print(term, file=vf)

    fetchIDF_cmd = \
        "sh ../idf_baseline/target/appassembler/bin/FetchTermIDF -index {} -vocabFile {}".\
            format(path_to_index, 'dataset.vocab')
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

def compute_overlap(questions, answers):
    """
    returns simple overlap between document pairs
    """
    overlap_scores = []
    for q, a in zip(questions, answers):
        q_terms = set(q.split())
        a_terms = set(a.split())
        common_terms = q_terms.intersection(a_terms)
        overlap = float(len(common_terms))/(len(q_terms) + len(a_terms))
        overlap_scores.append(overlap)
    return np.array(overlap_scores)

def compute_idf_weighted_overlap(questions, answers, idf_weights):
    """
    returns idf weighted overlap
    """
    overlap_scores = []
    for q, a in zip(questions, answers):
        q_terms = set(q.split())
        a_terms = set(a.split())
        common_terms = q_terms.intersection(a_terms)
        idf_weighted_overlap = np.sum([idf_weights[term] for term in list(common_terms)])
        idf_weighted_overlap /= (len(q_terms) + len(a_terms))
        overlap_scores.append(idf_weighted_overlap)
    return np.array(overlap_scores)


def set_external_features_as_per_paper(trainer, corpus_index=None):
    """
    computes external features as per the paper AND saves them into trainer
    """
    all_questions, all_answers = [], []
    for split in trainer.data_splits.keys():
        questions, answers, labels, max_q_len, max_a_len, default_ext_feats = \
            trainer.data_splits[split]
        all_questions.extend(questions)
        all_answers.extend(answers)

    all_data = set(all_questions + all_answers)
    print('corpus_index', corpus_index)
    idf_weights = get_qadata_only_idf(list(all_data)) if not corpus_index else \
                    get_source_corpus_idf(list(all_data), corpus_index)

    external_features = {}

    # NOTE: expected external features as per paper are
    # 1. overlap(q, a),
    # 2. idf_overlap(q, a),
    # 3. overlap(stopped(q), stopped(a)),
    # 4. idf_over(stopped(q), stopped(a))

    for split in trainer.data_splits.keys():
        questions, answers, labels, max_q_len, max_a_len, default_ext_feats = \
            trainer.data_splits[split]

        overlap = compute_overlap(questions, answers)
        idf_weighted_overlap = compute_idf_weighted_overlap(questions, answers, idf_weights)
        overlap_no_stopwords =\
            compute_overlap(stopped(questions), stopped(answers))
        idf_weighted_overlap_no_stopwords =\
            compute_idf_weighted_overlap(stopped(questions), stopped(answers), idf_weights)
        ext_feats = [np.array(feats) for feats in zip(overlap, idf_weighted_overlap,\
                    overlap_no_stopwords, idf_weighted_overlap_no_stopwords)]
        trainer.data_splits[split][-1] = ext_feats
        external_features[split] = ext_feats
    return external_features


def set_external_features_as_per_paper_and_stem(trainer, corpus_index=None):
    """
    computes external features as per the paper but performs stemming before computing IDF.
    features are saved into the trainer.data_splits
    """
    all_questions, all_answers = [], []
    for split in trainer.data_splits.keys():
        questions, answers, labels, max_q_len, max_a_len, default_ext_feats = \
            trainer.data_splits[split]
        all_questions.extend(questions)
        all_answers.extend(answers)

    all_data = set(all_questions + all_answers)

    # stem all words except stopwords to compute idf (required for feature number 2.)
    stoplist = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    def stem_non_stop_words(sentence):
        return ' '.join([stemmer.stem(word) if word not in stoplist else word \
                    for word in sentence.split()])
    all_but_stopwords_stemmed = [stem_non_stop_words(sentence) for sentence in list(all_data)]
    idf_weights = get_qadata_only_idf(all_but_stopwords_stemmed) if not corpus_index else \
                    get_source_corpus_idf(all_but_stopwords_stemmed, corpus_index)

    external_features = {}

    for split in trainer.data_splits.keys():
        questions, answers, labels, max_q_len, max_a_len, default_ext_feats = \
            trainer.data_splits[split]

        que_stem_all_but_stopwords = [stem_non_stop_words(que) for que in questions]
        ans_stem_all_but_stopwords = [stem_non_stop_words(ans) for ans in answers]

        overlap = compute_overlap(que_stem_all_but_stopwords, ans_stem_all_but_stopwords)
        idf_weighted_overlap = compute_idf_weighted_overlap(que_stem_all_but_stopwords,\
                                                            ans_stem_all_but_stopwords, idf_weights)

        que_stopped_stemmed = stemmed(stopped(questions))
        ans_stopped_stemmed = stemmed(stopped(answers))

        overlap_no_stopwords = compute_overlap(que_stopped_stemmed, ans_stopped_stemmed)
        idf_weighted_overlap_no_stopwords =\
            compute_idf_weighted_overlap(que_stopped_stemmed, ans_stopped_stemmed, idf_weights)
        ext_feats = [np.array(feats) for feats in zip(overlap, idf_weighted_overlap,\
                    overlap_no_stopwords, idf_weighted_overlap_no_stopwords)]
        trainer.data_splits[split][-1] = ext_feats
        external_features[split] = ext_feats
    return external_features