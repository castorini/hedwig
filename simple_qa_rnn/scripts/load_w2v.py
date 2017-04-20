import pickle

# load the pickled word vectors
w2v_pkl_path = "w2v_map_SQ.pkl"
print("loading word vectors from the pickle...")
w2v_map = None
with open(w2v_pkl_path, 'rb') as fh:
    w2v_map = pickle.load(fh)

print(list(w2v_map.keys()))
print(len(w2v_map.keys()))