import os

import numpy as np


def load_ranks(rank_file):
    score_dict = {}
    with open(rank_file, 'r') as f:
        for line in f:
            topic, _, docid, _, score, _ = line.split()
            if topic not in score_dict:
                score_dict[topic] = dict()
            score_dict[topic.strip()][docid.strip()] = float(score)
    return score_dict


def merge_ranks(old_ranks, new_ranks, topics):
    doc_ranks = dict()
    for topic in topics:
        missing_docids = list()
        old_scores = old_ranks[topic]
        new_scores = new_ranks[topic]
        if topic not in doc_ranks:
            doc_ranks[topic] = list(), list(), list()
        print("Processing documents in topic", topic)
        for docid, old_score in old_scores.items():
            try:
                new_score = new_scores[docid]
                doc_ranks[topic][0].append(docid)
                doc_ranks[topic][1].append(old_score)
                doc_ranks[topic][2].append(new_score)
            except KeyError:
                missing_docids.append(docid)
        print("Number of missing documents in topic %s: %d" % (topic, len(missing_docids)))
    return doc_ranks


def interpolate(old_scores, new_scores, alpha):
    s_min, s_max = min(old_scores), max(old_scores)
    old_score = (old_scores - s_min) / (s_max - s_min)
    s_min, s_max = min(new_scores), max(new_scores)
    new_score = (new_scores - s_min) / (s_max - s_min)
    score = old_score * (1 - alpha) + new_score * alpha
    return score


def rerank_alpha(doc_ranks, alpha, limit, filename, tag):
    filename = '%s_rerank_%0.1f.txt' % (filename, alpha)
    with open(os.path.join(filename), 'w') as f:
        print('Writing output for alpha', alpha)
        for topic in doc_ranks:
            docids, old_scores, new_scores = doc_ranks[topic]
            score = interpolate(np.array(old_scores), np.array(new_scores), alpha)
            sorted_score = sorted(list(zip(docids, score)), key=lambda x: -x[1])

            rank = 1
            for docids, score in sorted_score:
                f.write(f'{topic} Q0 {docids} {rank} {score} castor_{tag}\n')
                rank += 1
                if rank > limit:
                    break


def rerank(args, dataset):
    ret_ranks = load_ranks(args.ret_ranks)
    clf_ranks = load_ranks(args.clf_ranks)
    doc_ranks = merge_ranks(ret_ranks, clf_ranks, topics=dataset.TOPICS)

    filename = os.path.splitext(args.clf_ranks)[0]
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        rerank_alpha(doc_ranks, alpha, 10000, filename, tag="achyudh")