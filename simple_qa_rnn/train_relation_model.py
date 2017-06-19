import os
import sys
import time
import glob
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from relation_model import RelationPredictor
from args import get_args
import data
from utils.vocab import Vocab
from utils.read_data import *

args = get_args()
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(args.device)

# ---- helper methods ------
def evaluate_dataset_batch(data_set, model):
    n_total = data_set["size"]
    n_correct = 0
    num_batches = n_total // args.batch_size
    batch_indices = np.split(range(n_total),
                                range(args.batch_size, n_total, args.batch_size))
    model.eval()
    for batch_ix in range(num_batches):
        batch_questions = data_set["questions"][batch_indices[batch_ix]]
        batch_relations = data_set["rel_labels"][batch_indices[batch_ix]]
        inputs = Variable(read_text_tensor(batch_questions, word_vocab), volatile=True)
        targets = Variable(read_labels_tensor(batch_relations, rel_vocab), volatile=True)
        if args.cuda:
            inputs.data = inputs.data.cuda()
            targets.data = targets.data.cuda()
        scores = model(inputs)
        pred_score, pred_label_ix = torch.max(scores, dim=1) # check this properly
        pred_label_ix = pred_label_ix.view(args.batch_size)
        sum_correct = torch.sum(torch.eq(pred_label_ix, targets))
        n_correct += sum_correct.data[0]
    acc = n_correct / (num_batches * args.batch_size)
    return acc

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# ---- dataset paths ------
data_dir = "data/SimpleQuestions_v2/"
train_file = os.path.join(data_dir, "annotated_fb_data_train.txt")
val_file = os.path.join(data_dir, "annotated_fb_data_valid.txt")
test_file = os.path.join(data_dir, "annotated_fb_data_test.txt")

# ---- load GloVe embeddings ------
embed_pt_filepath = 'data/glove/glove.840B.300d.pt'
emb_w2i, emb_vecs = read_embedding(embed_pt_filepath)
emb_vocab = Vocab(emb_w2i)
emb_dim = emb_vecs.size()[1]

# ---- create dataset vocabulary and embeddings ------
vocab_pt_filepath = os.path.join(data_dir, "vocab.pt")
word2index_dict, rel2index_dict = torch.load(vocab_pt_filepath)

word_vocab = Vocab(word2index_dict)
word_vocab.add_pad_token("<PAD>")
word_vocab.add_unk_token("<UNK>")

rel_vocab = Vocab(rel2index_dict)

vocab_size = word_vocab.size
num_classes = len(rel2index_dict)
print('vocab size = {}'.format(vocab_size))
print('num classes = {}'.format(num_classes))

num_unk = 0
vecs = torch.FloatTensor(vocab_size, emb_dim)
for i in range(vocab_size):
    word = word_vocab.get_token(i)
    if emb_vocab.contains(word):
        vecs[i] = emb_vecs[emb_vocab.get_index(word)]
    elif word == word_vocab.pad_token:
        vecs[i].zero_()
    else:
        num_unk += 1
        vecs[i].uniform_(-0.05, 0.05)

print('unk vocab count = {}'.format(num_unk))
emb_vocab = None
emb_vecs = None

# ---- load datasets ------
print("loading train/val/test datasets...")
train_dataset = read_dataset(train_file, word_vocab, rel_vocab)
val_dataset = read_dataset(val_file, word_vocab, rel_vocab)
test_dataset = read_dataset(test_file, word_vocab, rel_vocab)
print('train_file: {}, num train = {}'.format(train_file, train_dataset["size"]))
print('val_file: {}, num dev   = {}'.format(val_file, val_dataset["size"]))
print('test_file: {}, num test  = {}'.format(test_file, test_dataset["size"]))


# ---- Define Model, Loss, Optim ------
config = args
config.vocab_size = vocab_size
config.d_out = num_classes
config.n_directions = 2 if config.birnn else 1
print(config)
model = RelationPredictor(config)
# initialize the embedding layer with the word vectors
model.embed.weight.data = vecs
if args.cuda:
    model.cuda()
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ---- Train Model ------
start = time.time()
best_val_acc = -1
iter = 0
header = '  Time Epoch Iteration     Loss   Train/Acc.   Val/Acc.'
print(header)
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>9.6f}'.split(','))
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>9.6f},{:9.6f},{:11.6f}'.split(','))

model.train()
for epoch in range(args.epochs):
    # shuffle the dataset and create batches (truncate the last batch if not of equal size)
    shuffled_indices = np.random.permutation(train_dataset["size"])
    num_batches = len(shuffled_indices) // args.batch_size
    batch_indices = np.split(shuffled_indices,
                                range(args.batch_size, len(shuffled_indices), args.batch_size))
    model.encoder.hidden = model.encoder.init_hidden()
    for batch_ix in range(num_batches):
        iter += 1
        batch_questions = train_dataset["questions"][batch_indices[batch_ix]]
        batch_relations = train_dataset["rel_labels"][batch_indices[batch_ix]]
        inputs = Variable( read_text_tensor(batch_questions, word_vocab) )
        targets = Variable( read_labels_tensor(batch_relations, rel_vocab) )
        if args.cuda:
            inputs.data = inputs.data.cuda()
            targets.data = targets.data.cuda()

        # clear out gradients and hidden states of the model
        model.zero_grad()
        model.encoder.hidden = repackage_hidden(model.encoder.hidden)

        # prepare inputs for LSTM model and run forward pass
        scores = model(inputs)

        # compute the loss, gradients, and update the parameters
        loss = loss_function(scores, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        # log at intervals
        if iter % args.dev_every == 0:
            train_acc = evaluate_dataset_batch(train_dataset, model)
            val_acc = evaluate_dataset_batch(val_dataset, model)
            model.train()
            print(dev_log_template.format(time.time()-start, epoch, iter, loss.data[0], train_acc, val_acc))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                snapshot_path = snapshot_prefix + '_valacc_{:6.4f}_trainacc{:6.4f}_iter_{}_model.pt'.format(val_acc, train_acc, iter)
                torch.save(model.state_dict(), snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        elif iter == 1 or iter % args.log_every == 0:
            print(log_template.format(time.time() - start, epoch, iter, loss.cpu().data[0]))

