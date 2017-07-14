import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torchtext import data

from model import RelationClassifier
from args import get_args
from simple_qa_relation import SimpleQaRelationDataset

# get the configuration arguments and set machine - GPU/CPU
args = get_args()
# set random seeds for reproducibility
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have CUDA but not using it.")
if torch.cuda.is_available() and args.cuda:
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)

# ---- prepare the dataset with Torchtext -----
questions = data.Field(lower=True)
relations = data.Field(sequential=False)

train, dev, test = SimpleQaRelationDataset.splits(questions, relations)

# build vocab for questions
questions.build_vocab(train, dev, test)

# load word vectors if already saved or else load it from start and save it
if os.path.isfile(args.vector_cache):
    questions.vocab.vectors = torch.load(args.vector_cache)
else:
    questions.vocab.load_vectors(wv_dir=args.data_cache, wv_type=args.word_vectors, wv_dim=args.d_embed)
    os.makedirs(os.path.dirname(args.vector_cache), exist_ok=True)
    torch.save(questions.vocab.vectors, args.vector_cache)

# build vocab for relations
relations.build_vocab(train, dev, test)

# BucketIterator buckets the examples according to length so less padding is needed
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=args.batch_size, device=args.gpu)
train_iter.repeat = False # do not repeat examples after finishing an epoch


# ---- define the model, loss, optim ------
config = args
config.n_embed = len(questions.vocab) # vocab. size / number of embeddings
config.d_out = len(relations.vocab)
config.n_cells = config.n_layers
# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2
print(config)

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage,location: storage.cuda(args.gpu))
else:
    model = RelationClassifier(config)
    if args.word_vectors:
        model.embed.weight.data = questions.vocab.vectors
        if args.cuda:
            model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# ---- train the model ------
iterations = 0
start = time.time()
best_dev_acc = -1
train_iter.repeat = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
os.makedirs(args.save_path, exist_ok=True)
print(header)

for epoch in range(args.epochs):
    train_iter.init_epoch()
    n_correct, n_total = 0, 0

    for batch_idx, batch in enumerate(train_iter):
        iterations += 1

        # switch model to training mode, clear gradient accumulators
        model.train(); optimizer.zero_grad()

        # forward pass
        answer = model(batch)

        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(answer, 1)[1].view(batch.relation.size()).data == batch.relation.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct/n_total

        # calculate loss of the network output with respect to training labels & backpropagate to compute gradients
        loss = criterion(answer, batch.relation)
        loss.backward()

        # clip the gradients (prevent exploding gradients) and update the weights
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_gradient)
        optimizer.step()

        # checkpoint model periodically
        if iterations % args.save_every == 0:
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.data[0], iterations)
            torch.save(model, snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:

            # switch model to evaluation mode
            model.eval(); dev_iter.init_epoch()

            # calculate accuracy on validation set
            n_dev_correct, dev_loss = 0, 0
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                 answer = model(dev_batch)
                 n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.relation.size()).data == dev_batch.relation.data).sum()
                 dev_loss = criterion(answer, dev_batch.relation)
            dev_acc = 100. * n_dev_correct / len(dev)

            print(dev_log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.data[0], dev_loss.data[0], train_acc, dev_acc))

            # update best valiation set accuracy
            if dev_acc > best_dev_acc:
                # found a model with better validation set accuracy
                best_dev_acc = dev_acc
                snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc, dev_loss.data[0], iterations)

                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        elif iterations % args.log_every == 0:

            # print progress message
            print(log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.data[0], ' '*8, n_correct/n_total*100, ' '*12))

