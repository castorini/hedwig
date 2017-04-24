#!/usr/bin/env python
# -*- coding: utf-8 -*-



from configurable import Configurable
class Example(Configurable):
  """

  """
  def __init__(self, sent, *args, **kwargs):
    super(Example, self).__init__(*args, **kwargs)
    self.length = len(sent)
    self.sent = None
    self.data = None
    # original word in this setting "TREC"
    # TODO: for different dataset, the original data will have different format
    # TODO: this data format is related to output. Leave for future work
    # self.feature = None
    # # Convert each of the features to one-hot representation: (n_word, n_feature)
    # self.target = None
    # # Convert each of the targets to one-hot representation: (n_word, n_target)
    if self.dataset_type == "TREC":
      self.data = {}
      self.sent = {}
      self.sent["words"] = sent[2:]
      self.sent["targets"] = sent[0]
    else:
      self.data = {}
      self.sent = {}
      self.sent["words"] = sent[1:]
      self.sent["targets"] = sent[0]

  def convert(self, vocabs):
    words, target = vocabs
    self.data["words"] = []
    self.data["targets"] = target[self.sent["targets"]]
    for word in self.sent["words"]:
      self.data["words"].append(words[word])



