import glob
import pickle

fnames = glob.glob('../datasets/SimpleQuestions_v2/annotated*.txt')
all_lines = []
for fname in fnames:
    with open(fname) as fin:
        for line in fin:
            all_lines.append(line.rstrip())

print("num of examples in train/val/test: {}".format(len(all_lines)))

all_relations = set()
for line in all_lines:
    rel = line.split("\t")[1]
    all_relations.add(rel)

print("num of relation types: {}".format(len(all_relations)))
print(all_relations)
print( ("www.freebase.com/music/release/region") in all_relations )

rel_to_ix = { rel:i for i, rel in enumerate(all_relations) }
# print(rel_to_ix)
# # dump the pickle
# print("dumping the w2v map pickle...")
# with open("../resources/rel_to_ix_SQ.pkl", 'wb') as fh:
#     pickle.dump(rel_to_ix, fh)