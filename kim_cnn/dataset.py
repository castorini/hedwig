#!/usr/bin/env python
# -*- coding: utf-8 -*-

from configurable import Configurable
from bucket import Bucket
from example import Example
from collections import Counter
from etc.kmeans import KMeans
import numpy as np
from etc.utils import clean_str, clean_str_sst

class Dataset(Configurable):
  """
  Dataset Class:
  - Store Data
  - Generate Minibatch
  - Padding
  """
  def __init__(self, filename, vocabs, *args, **kwargs):

    super(Dataset, self).__init__(*args, **kwargs)
    self._train = (filename == self.train_file)
    self.vocabs = vocabs
    self.buckets = [Bucket(self._config, name='Sents-%d' % i) for i in xrange(self.n_bkts)]
    self.id2position = []
    self.len2bkts = {}
    self.vocabs = vocabs
    self.reading_dataset(filename)
    self._finalize()

  def _finalize(self):
    for bucket in self.buckets:
      bucket.finalize()




  @property
  def n_bkts(self):
    if self._train:
      return super(Dataset, self).n_bkts
    else:
      return super(Dataset, self).n_valid_bkts


  def reading_dataset(self, filename):
    """
    :param filename:
    :return:
    """
    if self.dataset_type == 'SST-1' or self.dataset_type == 'SST-2':
      with open(filename) as f:
        buff = []
        for line_num, line in enumerate(f):
          line = clean_str_sst(line).split()
          if len(line) > 1:
            buff.append(line)
        self._process_buff(buff)
    else:
      with open(filename) as f:
        buff = []
        for line_num, line in enumerate(f):
          line = clean_str(line).split()
          if line:
            buff.append(line)
        self._process_buff(buff)
    return

  def _process_buff(self, buff):
    """
    :param buff:
    :return:
    """
    len_cntr = Counter()
    for sent in buff:
      len_cntr[len(sent)] += 1
    bkts_splits = KMeans(self.n_bkts, len_cntr).splits
    # Count the sents length
    # Use k-means to splits the sents into n_bkts parts

    # reset bucket size
    # map the lenth to bkts id
    prev_size = -1
    for bkt_idx, size in enumerate(bkts_splits):
      self.buckets[bkt_idx].set_size(size)
      self.len2bkts.update(zip(range(prev_size+1, size+1), [bkt_idx] * (size-prev_size)))
      prev_size = size
      # map all length from min to max to bkts id
      # some of lengths do not appear in the data set
    for sent in buff:
      # Add the sent to the specific bucket according to their length
      # Construct the sent into example first
      # And then push them into buckets
      bkt_idx = self.len2bkts[len(sent)]
      example = Example(sent, self._config)
      example.convert(self.vocabs)
      # save to bucket
      idx = self.buckets[bkt_idx].add(example)
      self.id2position.append((bkt_idx, idx))







  def minibatch(self, batch_size, input_idx, target_idx, shuffle=True):
    minibatches = []
    for bkt_idx, bucket in enumerate(self.buckets):
        if batch_size == 0:
          print("Please Specify the batch size")
          exit()
        else:
          n_tokens = len(bucket) * bucket.size
          n_splits = max(n_tokens // batch_size, 1)

        if shuffle:
          range_func = np.random.permutation
        else:
          range_func = np.arange
        arr_sp = np.array_split(range_func(len(bucket)), n_splits)
        for bkt_mb in arr_sp:
          minibatches.append((bkt_idx, bkt_mb))
        if shuffle:
          np.random.shuffle(minibatches)

        for bkt_idx, bkt_mb in minibatches:
          data = self.buckets[bkt_idx].data[bkt_mb]
          sents = self.buckets[bkt_idx].sents[bkt_mb]
          target = self.buckets[bkt_idx].target[bkt_mb]
          maxlen = np.max(np.sum(np.greater(data[:,:,0], 0), axis=1))
          # Do not use dynamic index like conll_index
          # For word, set 0 data = [(fea1, fea2, fea3), (fea1, fea2, fea3), ...]
          # For target, target = [(target1,), (target2,), ...]
          feed_dict = {
            'text' : data[:,:maxlen, input_idx],
            'label' : target[:, target_idx],
            'batch_size' : len(target)
          }
          yield feed_dict

  @property
  def sentsNum(self):
    return len(self.id2position)

