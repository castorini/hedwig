#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Adapted from Bi-Affine Parser code
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter

import numpy as np


# ***************************************************************
class KMeans(object):
  """"""

  # =============================================================
  def __init__(self, k, len_cntr):
    """"""

    # Error checking
    if len(len_cntr) < k:
      raise ValueError('Trying to sort %d data points into %d buckets' % (len(len_cntr), k))

    # Initialize variables
    self._k = k
    self._len_cntr = len_cntr
    self._lengths = sorted(self._len_cntr.keys())
    self._splits = []
    self._split2len_idx = {}
    self._len2split_idx = {}
    self._split_cntr = Counter()

    # Initialize the splits evenly
    lengths = []
    for length, count in self._len_cntr.items():
      lengths.extend([length] * count)
    lengths.sort()
    self._splits = [np.max(split) for split in np.array_split(lengths, self._k)]

    i = len(self._splits) - 1
    while i > 0:
      while self._splits[i - 1] >= self._splits[i] or self._splits[i - 1] not in self._len_cntr:
        self._splits[i - 1] -= 1
      i -= 1

    i = 1
    while i < len(self._splits) - 1:
      while self._splits[i] <= self._splits[i - 1] or self._splits[i] not in self._len_cntr:
        self._splits[i] += 1
      i += 1

    # Reindex everything
    split_idx = 0
    split = self._splits[split_idx]
    for len_idx, length in enumerate(self._lengths):
      count = self._len_cntr[length]
      self._split_cntr[split] += count
      if length == split:
        self._split2len_idx[split] = len_idx
        split_idx += 1
        if split_idx < len(self._splits):
          split = self._splits[split_idx]
          self._split_cntr[split] = 0
      elif length > split:
        raise IndexError()

    # Iterate
    old_splits = None
    # print('0) Initial splits: %s; Initial mass: %d' % (self._splits, self.get_mass()))
    i = 0
    while self._splits != old_splits:
      old_splits = list(self._splits)
      self.recenter()
      i += 1
    # print('%d) Final splits: %s; Final mass: %d' % (i, self._splits, self.get_mass()))

    self.reindex()
    return

  # =============================================================
  def recenter(self):
    """"""

    for split_idx in range(len(self._splits)):
      split = self._splits[split_idx]
      len_idx = self._split2len_idx[split]
      if split == self._splits[-1]:
        continue
      right_split = self._splits[split_idx + 1]

      # Try shifting the centroid to the left
      if len_idx > 0 and self._lengths[len_idx - 1] not in self._split_cntr:
        new_split = self._lengths[len_idx - 1]
        left_delta = self._len_cntr[split] * (right_split - new_split) - self._split_cntr[split] * (split - new_split)
        if left_delta < 0:
          self._splits[split_idx] = new_split
          self._split2len_idx[new_split] = len_idx - 1
          del self._split2len_idx[split]
          self._split_cntr[split] -= self._len_cntr[split]
          self._split_cntr[right_split] += self._len_cntr[split]
          self._split_cntr[new_split] = self._split_cntr[split]
          del self._split_cntr[split]

      # Try shifting the centroid to the right
      elif len_idx < len(self._lengths) - 2 and self._lengths[len_idx + 1] not in self._split_cntr:
        new_split = self._lengths[len_idx + 1]
        right_delta = self._split_cntr[split] * (new_split - split) - self._len_cntr[split] * (new_split - split)
        if right_delta <= 0:
          self._splits[split_idx] = new_split
          self._split2len_idx[new_split] = len_idx + 1
          del self._split2len_idx[split]
          self._split_cntr[split] += self._len_cntr[split]
          self._split_cntr[right_split] -= self._len_cntr[split]
          self._split_cntr[new_split] = self._split_cntr[split]
          del self._split_cntr[split]
    return

    # =============================================================

  def get_mass(self):
    """"""

    mass = 0
    split_idx = 0
    split = self._splits[split_idx]
    for len_idx, length in enumerate(self._lengths):
      count = self._len_cntr[length]
      mass += split * count
      if length == split:
        split_idx += 1
        if split_idx < len(self._splits):
          split = self._splits[split_idx]
    return mass

  # =============================================================
  def reindex(self):
    """"""

    self._len2split_idx = {}
    last_split = -1
    for split_idx, split in enumerate(self._splits):
      self._len2split_idx.update(dict(zip(range(last_split + 1, split), [split_idx] * (split - (last_split + 1)))))
    return

    # =============================================================

  def __len__(self):
    return self._k

  def __iter__(self):
    return (split for split in self.splits)

  def __getitem__(self, key):
    return self._splits[key]

  # =============================================================
  @property
  def splits(self):
    return self._splits

  @property
  def len2split_idx(self):
    return self._len2split_idx


# ***************************************************************
if __name__ == '__main__':
  """"""

  len_cntr = Counter()
  for i in range(10000):
    len_cntr[1 + int(10 ** (1 + np.random.randn()))] += 1
  print(len_cntr)
  kmeans = KMeans(10, len_cntr)
  print(kmeans.splits)