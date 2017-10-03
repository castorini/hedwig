from tqdm import tqdm
import array
import torch
import numpy as np

from argparse import ArgumentParser


def convert(fname, vocab):
    save_file = '{}.pt'.format(fname)
    stoi, vectors, dim = [], array.array('d'), None

    # TODO: fix by reading the .dimensions file
    vocab_size, dim = 2470719, 50
    W = np.memmap(fname, dtype=np.double, shape=(vocab_size, dim))


    print("Loading vectors from {}".format(fname))
    vectors = []
    for line in tqdm(W, total=len(W)):
        entry = line
        vectors.extend(entry)

    vectors = torch.Tensor(vectors).view(-1, dim)

    with open(vocab) as f:
        stoi = {word.strip():i for i, word in enumerate(f)}

    print('saving vectors to', save_file)
    torch.save((stoi, vectors, dim), save_file)

if __name__ == '__main__':
    parser = ArgumentParser(description='create word embedding')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--vocab', type=str, required=True)

    args = parser.parse_args()
    convert(args.input, args.vocab)
