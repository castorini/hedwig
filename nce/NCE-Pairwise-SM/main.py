import numpy as np
import random
import logging
import os

import torch
from torchtext import data

from args import get_args
from trec_dataset import TrecDataset
from utils.relevancy_metrics import get_map_mrr
from datasets.trecqa import TRECQA
from train import UnknownWordVecCache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

args = get_args()
config = args

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    logger.info("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    logger.info("Warning: You have Cuda but do not use it. You are using CPU for training")


if config.dataset == 'TREC':
    dataset_root = os.path.join(os.pardir, 'data', 'TrecQA/')
    train_iter, dev_iter, test_iter = TRECQA.iters(dataset_root, args.vector_cache, args.wordvec_dir, batch_size=args.batch_size, pt_file=True, device=args.gpu, unk_init=UnknownWordVecCache.unk)
else:
    logger.info("Unsupported dataset")
    exit()

config.target_class = 2
config.questions_num = len(TRECQA.TEXT_FIELD.vocab)
config.answers_num = len(TRECQA.TEXT_FIELD.vocab)

if args.cuda:
    model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
    model = torch.load(args.trained_model, map_location=lambda storage,location: storage)


def predict(test_mode, dataset_iter):
    model.eval()
    dataset_iter.init_epoch()
    qids = []
    predictions = []
    labels = []
    for dev_batch_idx, dev_batch in enumerate(dataset_iter):
        qid_array = np.transpose(dev_batch.id.cpu().data.numpy())
        true_label_array = np.transpose(dev_batch.label.cpu().data.numpy())
        output = model.convModel(dev_batch)
        scores = model.linearLayer(output)
        score_array = scores.cpu().data.numpy().reshape(-1)
        qids.extend(qid_array.tolist())
        predictions.extend(score_array.tolist())
        labels.extend(true_label_array.tolist())

    dev_map, dev_mrr = get_map_mrr(qids, predictions, labels)

    logger.info("{} {}".format(dev_map, dev_mrr))

# Run the model on the dev set
predict('dev', dataset_iter=dev_iter)

# Run the model on the test set
predict('test', dataset_iter=test_iter)
