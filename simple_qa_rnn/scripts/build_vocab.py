import os
import glob
import torch
import nltk
import string

# this method was copied from utils/read_data.py
def process_tokenize_text(text):
    punc_remover = str.maketrans('', '', string.punctuation)
    processed_text = text.lower().translate(punc_remover)
    tokens = nltk.word_tokenize(processed_text)
    return tokens

def build_vocab_SQ(data_dir):
    filepaths = glob.glob(os.path.join(data_dir, 'annotated*.txt'))
    print("reading filepaths: {}".format(filepaths))
    word_vocab = set()
    relation_vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                line_items = line.split("\t")
                # add relation
                relation = line_items[1]
                relation_vocab.add(relation)
                # add text
                qText = line_items[3]
                tokens = process_tokenize_text(qText)
                word_vocab |= set(tokens)

    word2index_dict = {word: i for i, word in enumerate(sorted(word_vocab))}  # word to index dictionary
    rel2index_dict = {relation: i for i, relation in enumerate(sorted(relation_vocab))}  # relation to index dictionary
    return (word2index_dict, rel2index_dict)


print("WARNING: This script is dataset specific. Please change it to fit your own dataset.")
data_dir = 'data/SimpleQuestions_v2/'
dst_path = os.path.join(data_dir, 'vocab.pt')
print("Building vocab for data in: {}".format(data_dir))
ret = build_vocab_SQ(data_dir) # ret = (word2index dict, relation2index dict)
print("saving word2index and answer2index dicts to {}".format(dst_path))
torch.save(ret, dst_path)
print("Done!")