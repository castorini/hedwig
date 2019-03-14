import numpy as np
import string
import pickle
from collections import defaultdict
from argparse import ArgumentParser

from nltk.stem.porter import PorterStemmer

def load_data(dname):
    stemmer = PorterStemmer()
    qids, questions, answers, labels = [], [], [], []
    print('Load folder ' + dname)
    with open(dname+'a.toks', encoding='utf-8') as f:
        for line in f:
            question = line.strip().split()
            question = [stemmer.stem(word) for word in question]
            questions.append(question)
    with open(dname+'b.toks', encoding='utf-8') as f:
        for line in f:
            answer = line.strip().split()
            answer_list = []
            for word in answer:
                try:
                    answer_list.append(stemmer.stem(word))
                except Exception as e:
                    print("couldn't stem the word:" + word)
            answers.append(answer_list)
    with open(dname+'id.txt', encoding='utf-8') as f:
        for line in f:
            qids.append(line.strip())
    with open(dname+'sim.txt', encoding='utf-8') as f:
        for line in f:
            labels.append(int(line.strip()))
    return qids, questions, answers, labels

def compute_overlap_features(questions, answers, word2df=None, stoplist=None):
    word2df = word2df if word2df else {}
    stoplist = stoplist if stoplist else set()
    feats_overlap = []
    for question, answer in zip(questions, answers):
        q_set = set([q for q in question if q not in stoplist])
        a_set = set([a for a in answer if a not in stoplist])
        word_overlap = q_set.intersection(a_set)
        if len(q_set) == 0 and len(a_set) == 0:
            overlap = 0
        else:
            overlap = float(len(word_overlap)) / (len(q_set) + len(a_set))

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

def compute_overlap_idx(questions, answers, stoplist, q_max_sent_length, a_max_sent_length):
    stoplist = stoplist if stoplist else []
    q_indices, a_indices = [], []
    for question, answer in zip(questions, answers):
        q_set = set([q for q in question if q not in stoplist])
        a_set = set([a for a in answer if a not in stoplist])
        word_overlap = q_set.intersection(a_set)

        q_idx = np.ones(q_max_sent_length) * 2
        for i, q in enumerate(question):
            value = 0
            if q in word_overlap:
                value = 1
            q_idx[i] = value
        q_indices.append(q_idx)

        a_idx = np.ones(a_max_sent_length) * 2
        for i, a in enumerate(answer):
            value = 0
            if a in word_overlap:
                value = 1
            a_idx[i] = value
        a_indices.append(a_idx)

    q_indices = np.vstack(q_indices).astype('int32')
    a_indices = np.vstack(a_indices).astype('int32')

    return q_indices, a_indices

def compute_dfs(docs):
    word2df = defaultdict(float)
    for doc in docs:
        for w in set(doc):
            word2df[w] += 1.0
    num_docs = len(docs)

    for w, value in word2df.items():
        word2df[w] = np.math.log(num_docs / value)   # bug feats fixed

    return word2df

if __name__ == '__main__':
    parser = ArgumentParser(description='create TrecQA/WikiQA dataset')
    parser.add_argument('--dir', help='path to the TrecQA|WikiQA data directory', default="../../Castor-data/TrecQA")
    args = parser.parse_args()

    stoplist = set([line.strip() for line in open('../../Castor-data/TrecQA/stopwords.txt', encoding='utf-8')])
    punct = set(string.punctuation)
    stoplist.update(punct)

    all_questions, all_answers, all_qids = [], [], []
    base_dir = args.dir

    if 'TrecQA' in base_dir:
        sub_dirs = ['train/', 'train-all/', 'raw-dev/', 'raw-test/']
    elif 'WikiQA' in base_dir:
        sub_dirs = ['train/', 'dev/', 'test/']
    else:
        print('Unsupported dataset')
        exit()

    for sub in sub_dirs:
        qids, questions, answers, labels = load_data(base_dir + sub)
        all_questions.extend(questions)
        all_answers.extend(answers)
        all_qids.extend(qids)

    seen = set()
    unique_questions = []
    for q, qid in zip(all_questions, all_qids):
        if qid not in seen:
            seen.add(qid)
            unique_questions.append(q)

    docs = all_answers + unique_questions
    word2dfs = compute_dfs(docs)
    pickle.dump(word2dfs, open("word2dfs.p", "wb"))

    q_max_sent_length = max(map(lambda x: len(x), all_questions))
    a_max_sent_length = max(map(lambda x: len(x), all_answers))

    for sub in sub_dirs:
        qids, questions, answers, labels = load_data(base_dir + sub)

        overlap_feats = compute_overlap_features(questions, answers, stoplist=None, word2df=word2dfs)
        overlap_feats_stoplist = compute_overlap_features(questions, answers, stoplist=stoplist, word2df=word2dfs)
        overlap_feats = np.hstack([overlap_feats, overlap_feats_stoplist])

        with open(base_dir + sub + 'overlap_feats.txt', 'w') as f:
            for i in range(overlap_feats.shape[0]):
                for j in range(4):
                    f.write(str(overlap_feats[i][j]) + ' ')
                f.write('\n')
