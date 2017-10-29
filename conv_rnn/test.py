import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils as utils

import data
import model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--input_file", default="saves/model.pt", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--gpu_number", default=0, type=int)
    args = parser.parse_args()

    conv_rnn = torch.load(args.input_file)
    if not args.no_cuda:
        torch.cuda.set_device(args.gpu_number)
        conv_rnn.cuda()
    _, _, test_set = data.SSTDataset.load_sst_sets("data")
    test_loader = utils.data.DataLoader(test_set, batch_size=len(test_set), collate_fn=conv_rnn.convert_dataset)

    conv_rnn.eval()
    for test_in, test_out in test_loader:
        scores = conv_rnn(test_in)
        n_correct = (torch.max(scores, 1)[1].view(-1).data == test_out.data).sum()
        accuracy = n_correct / len(test_set)
    print("Test set accuracy: {}".format(accuracy))

if __name__ == "__main__":
    main()
