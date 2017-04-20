import gensim
import glob
import pickle
import nltk
import string
import numpy as np

fnames = glob.glob('../datasets/SimpleQuestions_v2/annotated*.txt')
all_lines = []
for fname in fnames:
    with open(fname) as fin:
        for line in fin:
            all_lines.append(line.rstrip())

print("num of examples in train/val/test: {}".format(len(all_lines)))

max_sent_length = -1
for line in all_lines:
    qText = line.split("\t")[3]
    punc_remover = str.maketrans('', '', string.punctuation)
    processed_text = qText.lower().translate(punc_remover)
    tokens = nltk.word_tokenize(processed_text)
    if len(tokens) > max_sent_length:
        max_sent_length = len(tokens)

print("max_sent_length: {}".format(max_sent_length))



# # get all the words in the train, dev, val, test set
# all_words = set()
# for line in all_lines:
#     qText = line.split("\t")[3]
#     # process text: remove punctuations, lowercase
#     punc_remover = str.maketrans('', '', string.punctuation)
#     processed_text = qText.lower().translate(punc_remover)
#     tokens = nltk.word_tokenize(processed_text)
#     for tok in tokens:
#         all_words.add(tok)
#
# print("vocab. size: {}".format(len(all_words)))
#
# # word_to_ix = { word:i for i, word in enumerate(all_words) }
# # # dump the pickle
# # with open("../resources/word_to_ix_SQ.pkl", 'wb') as fh:
# #     pickle.dump(word_to_ix, fh)
#
# word_to_vector_map = {}
# w2v_path = "../resources/GoogleNews-vectors-negative300.bin.gz"
# # get their word vectors
# print("loading word vectors...")
# word_vectors = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
#
# # store in a dict
# found = 0
# random = 0
# for w in all_words:
#     # print(w)
#     try:
#         word_to_vector_map[w] = word_vectors[w]
#         found += 1
#     except:
#         word_to_vector_map[w] = np.random.uniform(low=0.0, high=1.0, size=300)
#         random += 1
#
# print("found: {}".format(found))
# print("random: {}".format(random))
#
# # dump the pickle
# print("dumping the w2v map pickle...")
# with open("../resources/w2v_map_SQ.pkl", 'wb') as fh:
#     pickle.dump(word_to_vector_map, fh)