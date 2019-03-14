import json
import os
import sys
from collections import Counter
import argparse
import random

import numpy as np
import torch
from nltk.tokenize import TreebankWordTokenizer
from torchtext import  data

from sm_cnn.external_features import compute_overlap, compute_idf_weighted_overlap, stopped
from sm_cnn.trec_dataset import TrecDataset
from sm_cnn.wiki_dataset import WikiDataset
from anserini_dependency.RetrieveSentences import RetrieveSentences
from sm_cnn import model

sys.modules['model'] = model

class SMModelBridge(object):

    def __init__(self, args):
        if not args.cuda:
            args.gpu = -1
        if torch.cuda.is_available() and args.cuda:
            print("Note: You are using GPU for training")
            torch.cuda.set_device(args.gpu)
            torch.cuda.manual_seed(args.seed)
        if torch.cuda.is_available() and not args.cuda:
            print("Warning: You have Cuda but do not use it. You are using CPU for training")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        self.QID = data.Field(sequential=False)
        self.QUESTION = data.Field(batch_first=True)
        self.ANSWER = data.Field(batch_first=True)
        self.LABEL = data.Field(sequential=False)
        self.EXTERNAL = data.Field(sequential=True, tensor_type=torch.FloatTensor, batch_first=True, use_vocab=False,
                              postprocessing=data.Pipeline(lambda arr, _, train: [float(y) for y in arr]))

        if 'TrecQA' in args.dataset:
            train, dev, test = TrecDataset.splits(self.QID, self.QUESTION, self.ANSWER, self.EXTERNAL, self.LABEL)
        elif 'WikiQA' in args.dataset:
            train, dev, test = WikiDataset.splits(self.QID, self.QUESTION, self.ANSWER, self.EXTERNAL, self.LABEL)
        else:
            print("Unsupported dataset")
            exit()

        self.QID.build_vocab(train, dev, test)
        self.QUESTION.build_vocab(train, dev, test)
        self.ANSWER.build_vocab(train, dev, test)
        self.LABEL.build_vocab(train, dev, test)

        if args.cuda:
            self.model = torch.load(args.model, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            self.model = torch.load(args.model, map_location=lambda storage, location: storage)

        self.gpu = args.gpu

    def parse(self, sentence):
        s_toks = TreebankWordTokenizer().tokenize(sentence)
        sentence = ' '.join(s_toks).lower()
        return sentence

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
            answer = answer.split('\t')[0]
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
            ext_feats = str(overlap[0]) + " " + str(idf_weighted_overlap[0]) + " " + \
                        str(overlap_no_stopwords[0]) + " " + str(idf_weighted_overlap_no_stopwords[0])


            fields = [('question', self.QUESTION), ('answer', self.ANSWER), ('ext_feat', self.EXTERNAL)]
            example = data.Example.fromlist([question, answer, ext_feats], fields)
            this_question = self.QUESTION.numericalize(self.QUESTION.pad([example.question]), self.gpu)
            this_answer = self.ANSWER.numericalize(self.ANSWER.pad([example.answer]), self.gpu)
            this_external = self.EXTERNAL.numericalize(self.EXTERNAL.pad([example.ext_feat]), self.gpu)
            self.model.eval()
            scores = self.model(this_question, this_answer, this_external)
            scores_sentences.append((scores[:, 2].cpu().data.numpy()[0].tolist(), answer))

        return scores_sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bridge Demo. Produces scores in trec_eval format",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help="the path to the saved model file")
    parser.add_argument('--dataset', help="the QA dataset folder {TrecQA|WikiQA}", default='../../Castor-data/TrecQA/')
    parser.add_argument("--index", help="Lucene index", required=True)
    parser.add_argument("--embeddings", help="Path of the word2vec index", default="")
    parser.add_argument("--topics", help="topics file", default="")
    parser.add_argument("--query", help="a single query", default="where was newton born ?")
    parser.add_argument("--hits", help="max number of hits to return", default=100)
    parser.add_argument("--scorer", help="passage scores", default="Idf")
    parser.add_argument("--k", help="top-k passages to be retrieved", default=1)
    parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0) # Use -1 for CPU
    parser.add_argument('--seed', type=int, default=3435)

    args = parser.parse_args()

    if not args.cuda:
        args.gpu = -1

    retrieveSentencesObj = RetrieveSentences(args)
    idf_json = retrieveSentencesObj.getTermIdfJSON()
    smmodel = SMModelBridge(args)

    train_set, dev_set, test_set = 'train', 'dev', 'test'
    if 'TrecQA' in args.dataset:
        train_set, dev_set, test_set = 'train-all', 'raw-dev', 'raw-test'

    for split in [dev_set, test_set]:
        outfile = open('bridge.{}.scores'.format(split), 'w')

        questions = [q.strip() for q in open(os.path.join(args.dataset, split, 'a.toks')).readlines()]
        answers = [q.strip() for q in open(os.path.join(args.dataset, split, 'b.toks')).readlines()]
        labels = [q.strip() for q in open(os.path.join(args.dataset, split, 'sim.txt')).readlines()]
        qids = [q.strip() for q in open(os.path.join(args.dataset, split, 'id.txt')).readlines()]

        qid_question = dict(zip(qids, questions))
        q_counts = Counter(questions)

        answers_offset = 0
        docid_counter = 0

        all_questions_answers = questions + answers
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
                    os.path.basename(args.dataset)
                ), file=outfile)
                docid_counter += 1
            if 'WikiQA' in args.dataset:
                docid_counter = 0

        outfile.close()
