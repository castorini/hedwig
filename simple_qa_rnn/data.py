import nltk
import string
import pickle
import numpy as np
import torch
from torch.autograd import Variable

def get_all_lines(data_filename):
    all_lines = []
    with open(data_filename) as fin:
        for line in fin:
            all_lines.append(line.rstrip())
    return all_lines

def create_rp_dataset(data_file):
    dataset = []
    all_lines = get_all_lines(data_file)
    for line in all_lines:
        line_split = line.split("\t")
        text = line_split[3]
        relation = line_split[1]
        dataset.append( (text, relation) )
    return np.array(dataset)

def tokenize_text(text):
    punc_remover = str.maketrans('', '', string.punctuation)
    processed_text = text.lower().translate(punc_remover)
    tokens = nltk.word_tokenize(processed_text)
    return tokens

def add_padding_tokens(text_tokens, max_length, pad_type='both', pad_token='<pad>'):
    num_pads = max_length - len(text_tokens)
    right_pad = int(num_pads / 2)
    left_pad = num_pads - right_pad
    if pad_type == "both":
        padded_tokens = [pad_token]*left_pad + text_tokens + [pad_token]*right_pad
    elif pad_type == "right":
        padded_tokens = text_tokens + [pad_token]*num_pads
    else:
        padded_tokens = [pad_token]*num_pads + text_tokens
    return padded_tokens

def load_map(pname):
    ret_map = None
    with open(pname, 'rb') as fh:
        ret_map = pickle.load(fh)
    return ret_map

def text_to_vector(text, w2v_map, pad=False, max_length=None):
    vec = []
    tokens = tokenize_text(text)
    if pad and (max_length != None):
        tokens = add_padding_tokens(tokens, max_length)
    for token in tokens:
        vec.append( w2v_map[token] )
    return np.array(vec)

def label_to_vector(label_ix, num_labels):
    # create one-hot vector label representation
    y_vec = np.zeros(num_labels, dtype=np.int32)
    y_vec[label_ix] = 1
    return y_vec

def create_tensorized_data(sentence, label, w2v_map, label_to_ix):
    # x.shape: |S| X |D| - sentence length can vary between examples, dimension is fixed
    x = text_to_vector(sentence, w2v_map)
    y = label_to_ix[label]
    inputs = Variable(torch.Tensor(x))
    targets = Variable(torch.LongTensor([y]))
    return inputs, targets

def create_tensorized_batch(batch, max_sent_length, w2v_map, label_to_ix):
    X = []
    y = []
    for sent, label in batch:
        X.append( text_to_vector(sent, w2v_map, pad=True, max_length=max_sent_length) )
        y.append( label_to_ix[label] )
    inputs = Variable(torch.Tensor(X))
    targets = Variable(torch.LongTensor(y))
    return inputs, targets