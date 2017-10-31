import time
import os
import random
import torch
import torch.nn as nn
import numpy as np
from torchtext import data
from args import get_args
from model import KimCNN
from SST1 import SST1Dataset
from utils import clean_str_sst

# Set default configuration in : args.py
args = get_args()

# Set random seed for reproducibility

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but not use it. You are using CPU for training.")
np.random.seed(args.seed)
random.seed(args.seed)

# Set up the data for training
# SST-1
if args.dataset == 'SST-1':
    TEXT = data.Field(batch_first=True, tokenize=clean_str_sst)
    LABEL = data.Field(sequential=False)
    train, dev, test = SST1Dataset.splits(TEXT, LABEL)

TEXT.build_vocab(train, min_freq=2)
LABEL.build_vocab(train)

if os.path.isfile(args.vector_cache):
    stoi, vectors, dim = torch.load(args.vector_cache)
    TEXT.vocab.vectors = torch.Tensor(len(TEXT.vocab), dim)
    for i, token in enumerate(TEXT.vocab.itos):
        wv_index = stoi.get(token, None)
        if wv_index is not None:
            TEXT.vocab.vectors[i] = vectors[wv_index]
        else:
            TEXT.vocab.vectors[i] = torch.Tensor.zero_(TEXT.vocab.vectors[i])
else:
    print("Error: Need word embedding pt file")
    exit(1)

#print('len(TEXT.vocab)', len(TEXT.vocab))
#print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)

config = args
config.target_class = len(LABEL.vocab)
config.words_num = len(TEXT.vocab)
config.embed_num = len(TEXT.vocab)


#print(config)
print("Dataset {}    Mode {}".format(args.dataset, args.mode))
print("VOCAB num",len(TEXT.vocab))
print("LABEL.target_class:", len(LABEL.vocab))
print("LABELS:",LABEL.vocab.itos)
print("Train instance", len(train))
print("Dev instance", len(dev))
print("Test instance", len(test))


if args.resume_snapshot:
    if args.cuda:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
    else:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
else:
    model = KimCNN(config)
    model.static_embed.weight.data.copy_(TEXT.vocab.vectors)
    model.non_static_embed.weight.data.copy_(TEXT.vocab.vectors)
    if args.cuda:
        model.cuda()
        print("Shift model to GPU")


parameter = filter(lambda p: p.requires_grad, model.parameters())
#for idx, p in enumerate(parameter):
#    print(idx, p)
optimizer = torch.optim.Adadelta(parameter, lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()
early_stop = False
best_dev_acc = 0
iterations = 0
iters_not_improved = 0
epoch = 0
start = time.time()
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(os.path.join(args.save_path, args.dataset), exist_ok=True)
print(header)


while True:
    if early_stop:
        print("Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, best_dev_acc))
        break
    epoch += 1
    train_iter.init_epoch()
    n_correct, n_total = 0, 0

    for batch_idx, batch in enumerate(train_iter):
        # Batch size : (Sentence Length, Batch_size)
        iterations += 1
        model.train(); optimizer.zero_grad()
        #print("Text Size:", batch.text.size())
        #print("Label Size:", batch.label.size())
        scores = model(batch)
        n_correct += (torch.max(scores, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total

        loss = criterion(scores, batch.label)
        loss.backward()

        optimizer.step()


        # Evaluate performance on validation set
        if iterations % args.dev_every == 1:
            # switch model into evalutaion mode
            model.eval(); dev_iter.init_epoch()
            n_dev_correct = 0
            dev_losses = []
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                scores = model(dev_batch)
                n_dev_correct += (torch.max(scores, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                dev_loss = criterion(scores, dev_batch.label)
                dev_losses.append(dev_loss.data[0])
            dev_acc = 100. * n_dev_correct / len(dev)
            print(dev_log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.data[0],
                                          sum(dev_losses) / len(dev_losses), train_acc, dev_acc))

            # Update validation results
            if dev_acc > best_dev_acc:
                iters_not_improved = 0
                best_dev_acc = dev_acc
                snapshot_path = os.path.join(args.save_path, args.dataset, args.mode+'_best_model.pt')
                torch.save(model, snapshot_path)
            else:
                iters_not_improved += 1
                if iters_not_improved >= args.patience:
                    early_stop = True
                    break

        if iterations % args.log_every == 1:
            # print progress message
            print(log_template.format(time.time() - start,
                                      epoch, iterations, 1 + batch_idx, len(train_iter),
                                      100. * (1 + batch_idx) / len(train_iter), loss.data[0], ' ' * 8,
                                      n_correct / n_total * 100, ' ' * 12))




















