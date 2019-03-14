import argparse
import os

from scipy.special import erf
from scipy.stats import truncnorm
import numpy as np

import data

def build_vector_cache(glove_filename, vec_cache_filename, vocab):
    print("Building vector cache...")
    with open(glove_filename) as f, open(vec_cache_filename, "w") as f2:
        for line in f:
            tok, vec = line.split(" ", 1)
            if tok in vocab:
                vocab.remove(tok)
                f2.write("{} {}".format(tok, vec))

def discrete_tnorm(a, b, tgt_loc, sigma=1, n_steps=100):
    def phi(zeta):
        return 1 / (np.sqrt(2 * np.pi)) * np.exp(-0.5 * zeta**2)
    def Phi(x):
        return 0.5 * (1 + erf(x / np.sqrt(2)))
    def tgt_loc_update(x):
        y1 = phi((a - x) / sigma)
        y2 = phi((b - x) / sigma)
        x1 = Phi((b - x) / sigma)
        x2 = Phi((a - x) / sigma)
        denom = x1 - x2 + 1E-4
        return y1 / denom - y2 / denom

    x = tgt_loc
    direction = np.sign(tgt_loc - (b - a))
    for _ in range(n_steps):
        x = tgt_loc - sigma * tgt_loc_update(x)
    tn = truncnorm((a - x) / sigma, (b - x) / sigma, loc=x, scale=sigma)
    rrange = np.arange(a, b + 1)
    pmf = tn.pdf(rrange)
    pmf /= np.sum(pmf)
    return pmf

def discrete_lerp(a, b, ground_truth):
    pmf = np.zeros(b - a + 1)
    c = int(np.ceil(ground_truth + 1E-8))
    f = int(np.floor(ground_truth))
    pmf[min(c - a, b - a)] = ground_truth - f
    pmf[f - a] = c - ground_truth
    return pmf

def smoothed_labels(truth, n_labels):
    return discrete_lerp(1, n_labels, truth)

def preprocess(filename, output_name="sim_sparse.txt"):
    print("Preprocessing {}...".format(filename))
    with open(filename) as f:
        values = [float(l.strip()) for l in f.readlines()]
    values = [" ".join([str(l) for l in smoothed_labels(v, 5)]) for v in values]
    with open(os.path.join(os.path.dirname(filename), output_name), "w") as f:
        f.write("\n".join(values))

def add_vocab(tok_filename, vocab):
    with open(tok_filename) as f:
        for line in f:
            vocab.update(line.strip().split())

def main():
    base_conf = data.Configs.base_config()
    sick_conf = data.Configs.sick_config()
    sick_folder = sick_conf.sick_data
    vocab = set()
    for name in ("train", "dev", "test"):
        preprocess(os.path.join(sick_folder, name, "sim.txt"))
        add_vocab(os.path.join(sick_folder, name, "a.toks"), vocab)
        add_vocab(os.path.join(sick_folder, name, "b.toks"), vocab)
    build_vector_cache(base_conf.wordvecs_file, sick_conf.sick_cache, vocab)

if __name__ == "__main__":
    main()
