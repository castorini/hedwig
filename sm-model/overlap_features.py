import sys
import re
import os
import numpy as np

import argparse

#from nltk.corpus import stopwords
from nltk.stem.porter import *
from collections import defaultdict

def load_data(dname):
  stemmer = PorterStemmer()
  qids, questions, answers, labels = [], [], [], []
  print dname
  with open(dname+'a.toks') as f:
    for line in f:
      line = unicode(line, errors='ignore')
      question = line.strip().split()
      question = [stemmer.stem(word) for word in question]
      questions.append(question)
  with open(dname+'b.toks') as f:
    for line in f:
      line = unicode(line, errors='ignore')
      answer = line.decode('utf-8').strip().split()
      answer = [stemmer.stem(word) for word in answer]
      answers.append(answer)
  with open(dname+'id.txt') as f:
    for line in f:
      qids.append(line.strip())
  with open(dname+'sim.txt') as f:
    for line in f:
      labels.append(int(line.strip())) 
  return qids, questions, answers, labels

def compute_overlap_features(questions, answers, word2df=None, stoplist=None):
  word2df = word2df if word2df else {}
  stoplist = stoplist if stoplist else set()
  feats_overlap = []
  for question, answer in zip(questions, answers):
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

    feats_overlap.append(np.array([
                         overlap,
                         df_overlap,
                         ]))
  return np.array(feats_overlap)

def compute_overlap_idx(questions, answers, stoplist, q_max_sent_length, a_max_sent_length):
  stoplist = stoplist if stoplist else []
  feats_overlap = []
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

    #### ERROR
    # a_idx = np.ones(a_max_sent_length) * 2
    # for i, q in enumerate(question):
    #   value = 0
    #   if q in word_overlap:

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
  for w, value in word2df.iteritems():
    word2df[w] /= np.math.log(num_docs / value)
  return word2df

if __name__ == '__main__':
  ap = argparse.ArgumentParser(description="compute overlap features for SM model")
  ap.add_argument("dataset", help="path/to/dataset-directory", default="../../data/TrecQA")
  ap.add_argument("--train_all", help="will generate overlap features for the train-all dataset", action="store_true")
  args = ap.parse_args()

  stoplist = set([line.strip() for line in open('stopwords.txt')])
  import string
  punct = set(string.punctuation)
  stoplist.update(punct) 
  #stoplist = None 
 
  all_questions, all_answers, all_qids = [], [], []
  base_dir = args.dataset
  # base_dir = '../../data/' + sys.argv[1] + '/'
  # sub_dirs = ['train/', 'raw-dev/', 'test.minimal/','test.complete/']
  # sub_dirs = ['train-all/', 'raw-dev/', 'raw-test/']
  sub_dirs = ['train/', 'clean-dev/', 'clean-test/']
  if args.train_all:
      sub_dirs = ['train-all/', 'raw-dev/', 'raw-test/']
      
  for sub in sub_dirs:
    qids, questions, answers, labels = load_data(base_dir+sub)
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
  print word2dfs.items()[:10]

  q_max_sent_length = max(map(lambda x: len(x), all_questions))
  a_max_sent_length = max(map(lambda x: len(x), all_answers))
  print 'q_max_sent_length', q_max_sent_length
  print 'a_max_sent_length', a_max_sent_length

  for sub in sub_dirs:
    print sub
    qids, questions, answers, labels = load_data(base_dir+sub)

    overlap_feats = compute_overlap_features(questions, answers, stoplist=None, word2df=word2dfs)
    overlap_feats_stoplist = compute_overlap_features(questions, answers, stoplist=stoplist, word2df=word2dfs)
    overlap_feats = np.hstack([overlap_feats, overlap_feats_stoplist])
    print overlap_feats[:3]
    print 'overlap_feats', overlap_feats.shape
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    print "Scaling overlap features"
    overlap_feats = scaler.fit_transform(overlap_feats)
    print overlap_feats[:3]    

    '''q_overlap_indices, a_overlap_indices = compute_overlap_idx(questions, answers, stoplist, q_max_sent_length, a_max_sent_length)
    print q_overlap_indices[:3], a_overlap_indices[:3]
    print 'q_overlap_indices', q_overlap_indices.shape
    print 'a_overlap_indices', a_overlap_indices.shape'''

    with open(base_dir+sub+'overlap_feats.txt', 'w') as f:
      for i in range(overlap_feats.shape[0]):
        for j in range(4):
          f.write(str(overlap_feats[i][j]) + ' ')
        f.write('\n')
    '''with open(base_dir+sub+'overlap_indices.txt', 'w') as f:
      for i in range(q_overlap_indices.shape[0]):
        for j in range(q_max_sent_length):
          f.write(str(q_overlap_indices[i][j]) + ' ')
        for j in range(a_max_sent_length):
          f.write(str(a_overlap_indices[i][j]) + ' ')
        f.write('\n')'''
