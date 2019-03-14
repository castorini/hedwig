from tqdm import tqdm
import torch

from gensim.models.keyedvectors import KeyedVectors
from argparse import ArgumentParser

def convert(fname, save_file):
    with open(fname, 'rb') as dim_file:
        vocab_size, dim = (int(x) for x in dim_file.readline().split())

    word_vectors = KeyedVectors.load_word2vec_format(fname, binary=True)

    print("Loading vectors from {}".format(fname))
    vectors = []
    for line in tqdm(word_vectors.syn0, total=len(word_vectors.syn0)):
        vectors.extend(line.tolist())
    vectors = torch.Tensor(vectors).view(-1, dim)

    stoi = {word.strip():voc.index for word, voc in word_vectors.vocab.items()}

    print('saving vectors to', save_file)
    torch.save((stoi, vectors, dim), save_file)

if __name__ == '__main__':
    parser = ArgumentParser(description='create word embedding')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='data/word2vec.trecqa.pt')

    args = parser.parse_args()
    convert(args.input, args.output)