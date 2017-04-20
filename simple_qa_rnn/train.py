import os
import sys
import time
import glob
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from model import BiLSTM
from util import get_args
import data

args = get_args()


# torch.cuda.set_device(args.gpu)


# ---- Helper Methods ------
def evaluate_dataset_batch(data_set, max_sent_length, model, w2v_map, label_to_ix):
    n_total = len(data_set)
    n_correct = 0
    num_batches = len(data_set) // args.batch_size
    batch_indices = np.split(range(n_total),
                                range(args.batch_size, n_total, args.batch_size))
    model.eval()
    for batch_ix in range(num_batches):
        batch = data_set[batch_indices[batch_ix]]
        inputs, targets = data.create_tensorized_batch(batch, max_sent_length, w2v_map, label_to_ix)
        scores = model(inputs)
        pred_label_ix = np.argmax(scores.data.numpy(), axis=1) # check this properly
        correct_label_ix = targets.data.numpy()
        n_correct += (pred_label_ix == correct_label_ix).sum()
    acc = n_correct / n_total
    return acc

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

#Load Datasets ------
train_file = "datasets/SimpleQuestions_v2/annotated_fb_data_train.txt"
val_file = "datasets/SimpleQuestions_v2/annotated_fb_data_valid.txt"
test_file = "datasets/SimpleQuestions_v2/annotated_fb_data_test.txt"

train_set = data.create_rp_dataset(train_file)
val_set = data.create_rp_dataset(val_file)
test_set = data.create_rp_dataset(test_file)
# train_set = train_set[:4]  # work with few examples first

# ---- Build Vocabulary ------
w2v_map = data.load_map("resources/w2v_map_SQ.pkl")
w2v_map['<pad>'] = np.zeros(300)
word_to_ix = data.load_map("resources/word_to_ix_SQ.pkl")
label_to_ix = data.load_map("resources/rel_to_ix_SQ.pkl")
vocab_size = len(word_to_ix)
num_classes = len(label_to_ix)
max_sent_length = 36  # set from the paper

# ---- Define Model, Loss, Optim ------
config = args
config.d_out = num_classes
config.n_directions = 2 if config.birnn else 1
print(config)
model = BiLSTM(config)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ---- Test Model ------
if args.test:
    print("Test Mode: loading pre-trained model and testing on test set...")
    # model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
    model.load_state_dict(torch.load(args.resume_snapshot))
    test_acc = evaluate_dataset_batch(test_set, max_sent_length, model, w2v_map, label_to_ix)
    print("Accuracy: {}".format(test_acc))
    sys.exit(0)


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
    shuffled_indices = np.random.permutation(len(train_set))
    num_batches = len(shuffled_indices) // args.batch_size
    batch_indices = np.split(shuffled_indices,
                                range(args.batch_size, len(shuffled_indices), args.batch_size))
    model.hidden = model.init_hidden()
    for batch_ix in range(num_batches):
        iter += 1
        batch = train_set[batch_indices[batch_ix]]
        inputs, targets = data.create_tensorized_batch(batch, max_sent_length, w2v_map, label_to_ix)
        # print("inputs size: {}".format(inputs.size()))
        # print("targets size: {}".format(targets.size()))

        # clear out gradients and hidden states of the model
        model.zero_grad()
        model.hidden = repackage_hidden(model.hidden)

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
            train_acc = evaluate_dataset_batch(train_set[:8000], max_sent_length, model, w2v_map, label_to_ix)
            val_acc = evaluate_dataset_batch(val_set, max_sent_length, model, w2v_map, label_to_ix)
            print(dev_log_template.format(time.time()-start, epoch, iter, loss.data[0], train_acc, val_acc))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                snapshot_path = snapshot_prefix + '_valacc_{:6.4f}__iter_{}_model.pt'.format(val_acc, iter)
                torch.save(model.state_dict(), snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        elif iter == 1 or iter % args.log_every == 0:
            print(log_template.format(time.time() - start, epoch, iter, loss.data[0]))

