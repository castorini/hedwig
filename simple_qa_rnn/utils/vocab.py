import torch

class Vocab(object):
    """
    A vocabulary object. Initialized from a file with one vocabulary token per line.
    Maps between vocabulary tokens and indices. If an UNK token is defined in the
    vocabulary, returns the index to this token if queried for an out-of-vocabulary
    token.
    """

    # def __init__(self, vocabpath):
    #     self.size = 0
    #     self.index = {}
    #     self.tokens = {}
    #
    #     with open(vocabpath, 'r') as f:
    #         for line in f:
    #             word = line.rstrip()
    #             self.tokens[self.size] = word
    #             self.index[word] = self.size
    #             self.size += 1
    #     # automatically add unknown token
    #     # self.add_unk_token("<UNK>")

    def __init__(self, word2index):
        self.index = word2index
        self.size = len(word2index)
        self.tokens = {index: word for word, index in word2index.items()}
        # self.add_unk_token("<UNK>")

    def contains(self, word):
        return word in self.index.keys()

    def add(self, word):
        if not self.contains(word):
            self.tokens[self.size] = word
            self.index[word] = self.size
            self.size += 1

    def add_unk_token(self, token):
        self.unk_token = token
        self.add(token)
        self.unk_index = self.index[token]

    def add_pad_token(self, token):
        self.pad_token = token
        self.add(token)
        self.pad_index = self.index[token]

    def get_index(self, word):
        if self.contains(word):
            return self.index[word]
        else:
            print("{} - word not found in vocab. returning unk_index".format(word))
            return self.unk_index

    def get_token(self, index):
        if index < 0 or index >= self.size:
            raise IndexError('index {} out of bounds'.format(index))
        return self.tokens[index]

    def map(self, tokens):
        N = len(tokens)
        out = torch.IntTensor(N)
        for i in range(N):
            out[i] = self.index(tokens[i])
        return out

