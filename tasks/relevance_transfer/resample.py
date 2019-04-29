# MIT License
#
# Copyright (c) 2018 Ming
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modified by: Achyudh Keshav Ram
# On: 7th Feb 2019

import torch
import torch.utils.data


class ImbalancedDatasetSampler:
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, labels, indices=None, num_samples=None):

        # All elements in the dataset will be considered if indices is None
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # Compute distribution of classes in the dataset
        self.labels = labels
        label_to_count = dict()
        for idx in self.indices:
            label = self.labels[idx].item()
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # Compute weight for each sample
        weights = [1.0 / label_to_count[self.labels[idx].item()]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def get_indices(self):
        return list(self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
