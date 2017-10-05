import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

import data
import model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--input_file", default="saves/model.pt", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--gpu_number", default=0, type=int)
    args = parser.parse_args()

    model.set_seed(5, no_cuda=args.no_cuda)
    data_loader = data.SSTDataLoader(args.data_dir)
    conv_rnn = torch.load(args.input_file)
    if not args.no_cuda:
        torch.cuda.set_device(args.gpu_number)
        conv_rnn.cuda()
    _, _, test_set = data_loader.load_sst_sets()

    conv_rnn.eval()
    test_in, test_out = conv_rnn.convert_dataset(test_set)
    scores = conv_rnn(test_in)
    n_correct = (torch.max(scores, 1)[1].view(len(test_set)).data == test_out.data).sum()
    accuracy = n_correct / len(test_set)
    print("Test set accuracy: {}".format(accuracy))

if __name__ == "__main__":
    main()
