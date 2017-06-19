import json
import os

rel_dir = 'data/dataset-factoid-webquestions/d-freebase-rp/'
text_dir = 'data/dataset-factoid-webquestions/main/'
out_dir = 'data/webquestions-custom/relation-prediction'
fname = 'val.json'
rel_filename = os.path.join(rel_dir, fname)
text_filename = os.path.join(text_dir, fname)
out_filename = os.path.join(out_dir, fname)

text_entries = []
with open(text_filename) as fin:
        data = json.load(fin)
        text_entries.extend(data)

rel_entries = []
with open(rel_filename) as fin:
        data = json.load(fin)
        rel_entries.extend(data)


print("num of examples: {}".format(len(text_entries)))
assert len(text_entries) == len(rel_entries)

outfile = open(out_filename, 'w')
count = 0
for text_entry, rel_entry in zip(text_entries, rel_entries):
    assert text_entry.get('qId') == rel_entry.get('qId')
    qID = text_entry.get('qId')
    qText = text_entry.get('qText')
    relPaths = rel_entry.get('relPaths')
    for relPath in relPaths:
        relations = relPath[0]
        for rel in relations:
            count += 1
            outfile.write("{}\t{}\t{}\n".format(qID, rel, qText))

print("count: {}".format(count))
outfile.close()
