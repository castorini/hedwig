import glob
import json

fnames = glob.glob('dataset-factoid-webquestions/d-freebase-rp/*.json')
all_entries = []
for fname in fnames:
    with open(fname) as fin:
        data = json.load(fin)
        all_entries.extend(data)

print("num of examples in train/val/test: {}".format(len(all_entries)))

all_relations = set()
for entry in all_entries:
    qID = entry.get('qId')
    relPaths = entry.get('relPaths')
    for relPath in relPaths:
        relations = relPath[0]
        for rel in relations:
            all_relations.add(rel)

print("num of relation types: {}".format(len(all_relations)))
print(all_relations)
print( ("/people/marriage/spouse") in all_relations )