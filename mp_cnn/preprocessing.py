"""
Preprocessing utilities such as preparing embeddings.
"""
import os

import numpy as np
import torch
import torch.nn as nn


def get_vocab(data_dir):
    """
    Get vocabulary as set of words.
    """
    vocab = set()
    with open(os.path.join(data_dir, 'vocab-cased.txt'), 'r') as f:
        for line in f:
            word = line.rstrip()
            vocab.add(word)
    return vocab


def get_embedding_index(vocab, glove_file):
    """
    Get dictionary mapping word to its word vector.
    """
    embedding_index = {}
    with open(glove_file, 'r') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word not in vocab or len(parts) != 301:
                continue
            vec = np.asarray(parts[1:], dtype='float32')
            embedding_index[word] = vec
    return embedding_index


def get_glove_embedding(glove_file, data_dir):
    """
    Get embedding for the words in the data set.
    """
    vocab = get_vocab(data_dir)
    embedding_index = get_embedding_index(vocab, glove_file)
    word_index = {w: i for i, w in enumerate(embedding_index.keys())}

    embedding_matrix = np.zeros((len(word_index), 300))
    for word, i in word_index.items():
        embedding_matrix[i] = embedding_index.get(word)

    embedding_tensor = torch.from_numpy(embedding_matrix)
    embedding = nn.Embedding(len(embedding_index), 300)
    embedding.weight = nn.Parameter(embedding_tensor)
    return word_index, embedding

