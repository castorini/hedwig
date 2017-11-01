"""
Utilities to compute IDF scores.
"""
from collections import defaultdict

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
import numpy as np


def get_pairwise_word_to_doc_freq(sent_list_1, sent_list_2):
    """
    Get pairwise word to document frequency.
    For index i, if sentence i in sent_list_1 and sentence i in sent_list_2 both
    container word w, then w is counted only once.
    Returns a dictionary mapping words to number of sentence pairs the word appears in.
    """
    word_to_doc_cnt = defaultdict(int)

    for s1, s2 in zip(sent_list_1, sent_list_2):
        unique_tokens = set(s1) | set(s2)
        for t in unique_tokens:
            word_to_doc_cnt[t] += 1

    return word_to_doc_cnt


def get_pairwise_overlap_features(sent_list_1, sent_list_2, word_to_doc_cnt):
    """
    Get overlap, idf weighted overlap, overlap excluding stopwords, and idf weighted overlap excluding stopwords.
    """
    stoplist = set(stopwords.words('english'))
    num_docs = len(sent_list_1)
    overlap_feats = []

    for s1, s2 in zip(sent_list_1, sent_list_2):
        tokens_a_set, tokens_b_set = set(s1), set(s2)
        intersect = tokens_a_set & tokens_b_set
        overlap = len(intersect) / (len(tokens_a_set) + len(tokens_b_set))
        idf_intersect = sum(np.math.log(num_docs / word_to_doc_cnt[w]) for w in intersect)
        idf_weighted_overlap = idf_intersect / (len(tokens_a_set) + len(tokens_b_set))

        tokens_a_set_no_stop = set(w for w in s1 if w not in stoplist)
        tokens_b_set_no_stop = set(w for w in s2 if w not in stoplist)
        intersect_no_stop = tokens_a_set_no_stop & tokens_b_set_no_stop
        overlap_no_stop = len(intersect_no_stop) / (len(tokens_a_set_no_stop) + len(tokens_b_set_no_stop))
        idf_intersect_no_stop = sum(np.math.log(num_docs / word_to_doc_cnt[w]) for w in intersect_no_stop)
        idf_weighted_overlap_no_stop = idf_intersect_no_stop / (len(tokens_a_set_no_stop) + len(tokens_b_set_no_stop))
        overlap_feats.append([overlap, idf_weighted_overlap, overlap_no_stop, idf_weighted_overlap_no_stop])

    return overlap_feats
