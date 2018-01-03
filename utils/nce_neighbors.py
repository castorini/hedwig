import random
import numpy as np
import heapq
import operator

import torch
from torchtext import data

# get the nearest negative samples to the positive sample by computing the feature difference
def get_nearest_neg_id(pos_feature, neg_dict, distance="cosine", k=1, weight=False):
    dis_list = []
    pos_feature = pos_feature.data.cpu().numpy()
    pos_feature_norm = pos_feature / np.sqrt(sum(pos_feature ** 2))
    neg_list = []
    for key in neg_dict:
        if distance == "l2":
            dis = np.sqrt(np.sum((np.array(pos_feature) - neg_dict[key]["feature"]) ** 2))
        elif distance == "cosine":
            neg_feature = np.array(neg_dict[key]["feature"])
            feat_norm = neg_feature / np.sqrt(sum(neg_feature ** 2))
            dis = 1 - feat_norm.dot(pos_feature_norm)
        dis_list.append(dis)
        neg_list.append(key)

    k = min(k, len(neg_dict))
    min_list = heapq.nsmallest(k, enumerate(dis_list), key=operator.itemgetter(1))
    # find the corresponding neg id
    min_id_list = [neg_list[x[0]] for x in min_list]
    if weight:
        min_id_score = [1 - x[1] for x in min_list]
        return min_id_list, min_id_score
    else:
        return min_id_list

# get the negative samples randomly
def get_random_neg_id(q2neg, qid_i, k=8):
    # question 1734 in TrecQA has only one positive answer and no negative answer
    if qid_i not in q2neg:
        return []
    k = min(k, len(q2neg[qid_i]))
    ran = random.sample(q2neg[qid_i], k)
    return ran

# pack the lists of question/answer/ext_feat into a torchtext batch
def get_batch(question, answer, ext_feat, size):
    new_batch = data.Batch()
    new_batch.batch_size = size
    setattr(new_batch, "sentence_2", torch.stack(answer))
    setattr(new_batch, "sentence_1", torch.stack(question))
    setattr(new_batch, "ext_feats", torch.stack(ext_feat))
    return new_batch