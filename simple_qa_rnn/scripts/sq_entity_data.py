import glob
import os

fnames = glob.glob('../data/SimpleQuestions_v2/annotated*.txt')
print(fnames)
all_entities = set()
for fname in fnames:
    with open(fname) as fin:
        for line in fin:
            entity = line.split("\t")[0]
            # entity: www.freebase.com/m/0f3xg_ --> m.0f3xg_
            if entity.startswith("www.freebase.com/"):
                entity = entity[17:].replace("/", ".")
            all_entities.add(entity)

print("num of entities in train/val/test: {}".format(len(all_entities)))

names_map = {}
names_file = "../data/freebase/names-map-0.ttl"
with open(names_file) as fin:
    for line in fin:
        id = line.split("\t")[0][3:]
        name = line.split("\t")[2].rstrip()
        if name.endswith("\"@en."):
            name = name[1:-5]
        if id in all_entities:
            names_map[id] = name

print("num of FB entities in the map: {}".format(len(names_map)))

found = len(names_map)
print("found: {}".format(found))
print("notfound: {}".format(len(all_entities) - found))

outfile = open("names_map.tsv", 'w')
for id, name in names_map.items():
    outfile.write("{}\t{}\n".format(id, name))

outfile.close()
print("done")