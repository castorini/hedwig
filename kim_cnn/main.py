import sys
import random
import numpy as np
import torch
from torchtext import data
from args import get_args

from datasets.sst import SST1

args = get_args()
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but do not use it. You are using CPU for training")
np.random.seed(args.seed)
random.seed(args.seed)

if not args.trained_model:
    print("Error: You need to provide a option 'trained_model' to load the model")
    sys.exit(1)

if args.dataset == 'SST-1':
    train_iter, dev_iter, test_iter = SST1.iters(args.data_dir, args.word_vectors_file, args.word_vectors_dir, batch_size=args.batch_size, device=args.gpu)

config = args
config.target_class = train_iter.dataset.NUM_CLASSES
config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)
config.embed_num = len(train_iter.dataset.TEXT_FIELD.vocab)

if args.cuda:
    model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
    model = torch.load(args.trained_model, map_location=lambda storage,location: storage)


def predict(dataset_iter, dataset_name):
    print("Dataset: {}".format(dataset_name))
    model.eval()
    dataset_iter.init_epoch()

    n_correct = 0
    for data_batch_idx, data_batch in enumerate(dataset_iter):
        scores = model(data_batch)
        n_correct += (torch.max(scores, 1)[1].view(data_batch.label.size()).data == data_batch.label.data).sum()

    print("no. correct {} out of {}".format(n_correct, len(dataset_iter.dataset.examples)))
    accuracy = 100. * n_correct / len(dataset_iter.dataset.examples)
    print("{} accuracy: {:8.6f}%".format(dataset_name, accuracy))

# Run the model on the dev set
predict(dataset_iter=dev_iter, dataset_name="valid")

# Run the model on the test set
predict(dataset_iter=test_iter, dataset_name="test")
