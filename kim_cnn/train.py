import time
import os
import random
import torch
import torch.nn as nn
import numpy as np

from datasets.sst import SST1
from args import get_args
from model import KimCNN

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

# Set up the data for training SST-1
if args.dataset == 'SST-1':
    train_iter, dev_iter, test_iter = SST1.iters(args.data_dir, args.word_vectors_file, args.word_vectors_dir, batch_size=args.batch_size, device=args.gpu)

config = args
config.target_class = train_iter.dataset.NUM_CLASSES
config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)
config.embed_num = len(train_iter.dataset.TEXT_FIELD.vocab)

print("Dataset {}    Mode {}".format(args.dataset, args.mode))
print("VOCAB num",len(train_iter.dataset.TEXT_FIELD.vocab))
print("LABEL.target_class:", train_iter.dataset.NUM_CLASSES)
print("Train instance", len(train_iter.dataset))
print("Dev instance", len(dev_iter.dataset))
print("Test instance", len(test_iter.dataset))

if args.resume_snapshot:
    if args.cuda:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
    else:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
else:
    model = KimCNN(config)
    model.static_embed.weight.data.copy_(train_iter.dataset.TEXT_FIELD.vocab.vectors)
    model.non_static_embed.weight.data.copy_(train_iter.dataset.TEXT_FIELD.vocab.vectors)
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
        iterations += 1
        model.train()
        optimizer.zero_grad()
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
            model.eval()
            dev_iter.init_epoch()
            n_dev_correct = 0
            dev_losses = []
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                scores = model(dev_batch)
                n_dev_correct += (torch.max(scores, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                dev_loss = criterion(scores, dev_batch.label)
                dev_losses.append(dev_loss.item())
            dev_acc = 100. * n_dev_correct / len(dev_iter.dataset)
            print(dev_log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.item(),
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
                                      100. * (1 + batch_idx) / len(train_iter), loss.item(), ' ' * 8,
                                      n_correct / n_total * 100, ' ' * 12))
