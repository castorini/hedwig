#!/usr/bin/env python
# -*- coding: utf-8 -*-

from configurable import Configurable
from collections import Counter
import numpy as np

from etc.utils import clean_str, clean_str_sst

class Vocab(Configurable):
  """
  Vocab for
  - id-str converting
  - word embedding / lookup
  """
  def __init__(self, vocab_file, *args, **kwargs):
    """
    vocab_file: the name of the vocab_file
    args: it should be self._config
    kwargs: some options for Vocab, before super, they should be pop out
    because these are not options for the global settings
    """
    self.vocab_file = vocab_file
    load_embed_file = kwargs.pop('load_embed_file', False)
    self.lower_case = kwargs.pop('lower_case', False)
    super(Vocab, self).__init__(*args, **kwargs)

    self.SPECIAL_TOKENS = ('<PAD>', '<UNK>')
    self.START_IDX = len(self.SPECIAL_TOKENS)
    self.PAD, self.UNK = range(self.START_IDX)
    self.pretrained_embeddings = None
    self._count = Counter() # Count the number of vocab

    self._str2idx = dict(zip(self.SPECIAL_TOKENS, range(self.START_IDX)))
    self._idx2str = dict(zip(range(self.START_IDX), self.SPECIAL_TOKENS))

    self._str2embed = {}
    self._embed2str = {}

    self.add_train_file()
    self.save_vocab_file()
    if load_embed_file:
      self.load_embed_file()



  def add_train_file(self):
    if self.dataset_type == 'TREC':
      with open(self.train_file) as f:
        for line_num, line in enumerate(f):
          line = clean_str(line).split()
          if line:
            if self.name == 'Targets':
              self.add(line[0])
            if self.name == 'Words':
              for word in line[2:]:
                self.add(word)
    else:
      with open(self.train_file) as f:
        for line_num, line in enumerate(f):
          line = clean_str(line).split()
          if line:
            if self.name == 'Targets':
              self.add(line[0])
            if self.name == 'Words':
              for word in line[1:]:
                self.add(word)

    self.index_vocab()

  def add(self, item):
    if self.lower_case:
      item = item.lower()

    self._count[item] += 1
    return

  def index_vocab(self):
    """
    Sorted the vocabs by frequency and assign id to them
    Process:
    - Get all the words with same frequency from the Counter
    - Sort those words
    - Assign ID to them
    - Go back to first step
    """
    cur_idx = self.START_IDX
    buff = []
    for word_and_count in self._count.most_common():
      if (not buff) or (buff[-1][1]==word_and_count[1]):
        buff.append(word_and_count)
      else:
        buff.sort()
        for word, count in buff:
          if count >= self.min_occur_count and (word not in self._str2idx):
            self._str2idx[word] = cur_idx
            self._idx2str[cur_idx] = word
            cur_idx += 1
        buff = [word_and_count]
    buff.sort()
    for word, count in buff:
      if count >= self.min_occur_count and word not in self._str2idx:
        self._str2idx[word] = cur_idx
        self._idx2str[cur_idx] = word
        cur_idx += 1
    return


  def save_vocab_file(self):
    """
    save the words on the file
    """
    with open(self.vocab_file, "w") as f:
      for word_and_count in self._count.most_common():
        f.write('%s\t%d\n' %(word_and_count))
    return

  def load_embed_file(self):
    self._str2embed = dict(zip(self.SPECIAL_TOKENS, range(self.START_IDX)))
    self._embed2str = dict(zip(range(self.START_IDX), self.SPECIAL_TOKENS))
    embeds = [[0] * self.words_dim, [0] * self.words_dim]
    with open(self.embed_file) as f:
      cur_idx = self.START_IDX
      for line_num, line in enumerate(f):
        line = line.strip().split()
        if line:
          try:
            if self.dataset_type != 'SST-1' or self.dataset_type != 'SST-2':
              self._str2embed[clean_str(line[0])] = cur_idx
              self._embed2str[cur_idx] = clean_str(line[0])
            else:
              self._str2embed[clean_str_sst(line[0])] = cur_idx
              self._embed2str[cur_idx] = clean_str_sst(line[0])
            embeds.append(line[1:])
            cur_idx += 1
          except:
            raise ValueError('The embedding file is misformatted at line %d' % (line_num+1))
    # Randomly initialize the pre-trained vector for those words not in pre-train-file
    for word in self._str2idx.keys():
      if word not in self._str2embed.keys():
        self._str2embed[word] = cur_idx
        self._embed2str[cur_idx] = word
        embeds.append(list(np.random.uniform(-1, 1, self.words_dim)))
        cur_idx += 1
    self.pretrained_embeddings = np.array(embeds, dtype=np.float64)
    del embeds
    return


  def __getitem__(self, key):
    if isinstance(key, basestring):
      # Convert the lower case
      if self.lower_case:
        key = key.lower()
      if self.pretrained_embeddings is not None:
        return (self._str2idx.get(key, self.UNK), self._str2embed.get(key, self.UNK))
      else:
        return (self._str2idx.get(key, self.UNK),)

  def __len__(self):
    return len(self._str2idx)

  @property
  def embeds_size(self):
    return len(self._embed2str)
