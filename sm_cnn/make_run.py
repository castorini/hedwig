import argparse
import os
from model import QAModel
from train import Trainer
import utils

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Makes a run in trec_eval run format, given a model and a train|dev|test set" )
    ap.add_argument('model')
    ap.add_argument('--word_embeddings_cache',
                    default='../../data/word2vec/aquaint+wiki.txt.gz.ndim=50.cache')
    ap.add_argument('--paper-ext-feats', action="store_true", \
        help="external features as per the paper")
    ap.add_argument('dataset_folder', help="the QA dataset folder",
                    choices=['../../data/TrecQA', '../../data/WikQA'])
    ap.add_argument('set_split', help="train, dev or test split as the data_folder")
    ap.add_argument("batch_size", help="the number of pairs to compare in each batch.\
                     should be same as during training")
    ap.add_argument('out_scorefile', help='output file in trec_eval format')


    args = ap.parse_args()

    vocab_size, vec_dim = utils.load_embedding_dimensions(args.word_embeddings_cache)

    trained_model = QAModel.load(args.model)
    trained_model.no_ext_feats = True
    evaluator = Trainer(trained_model, 0, 0, False, vec_dim) # 0, 0, False are dummy arguments
    evaluator.load_input_data(args.dataset_folder, args.word_embeddings_cache,
                              None, None, args.set_split,
                              True if args.ext_feats else False)
    test_scores = evaluator.test(args.set_split, args.batch_size)

    questions, sentences, labels, vocab, maxlen_q, maxlen_s, ext_feats = \
            evaluator.data_splits[args.set_split]

    qids = [id.strip() for id in open(os.path.join(args.dataset_folder, args.set_split, 'id.txt'))\
            .readlines()]

    with open(args.out_scorefile, 'w') as outf:
        old_qid = 0
        docid_c = 0
        for i in range(len(qids)):
            if qids[i] != old_qid and args.dataset_folder.endswith('WikiQA'):
                docid_c = 0
                old_qid = qids[i]
            print('{} 0 {} 0 {} {}'.format(qids[i], docid_c, test_scores[i],
                                           os.path.basename(args.model)),
                  file=outf)
            docid_c += 1
