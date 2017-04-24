#!/usr/bin/env python
# -*- coding: utf-8 -*-

from configurable import Configurable
from vocab import Vocab
from dataset import Dataset
import os
import sys
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class cnnTextNetwork(Configurable):
  """
  Network class
  - build the vocabulary
  - build the dataset
  - control the training
  - control the validation and testing
  - save the model and store the best result
  """

  def __init__(self, option, model, *args, **cargs):

    '''check args?'''
    super(cnnTextNetwork, self).__init__(*args, **cargs)
    if not os.path.isdir(self.save_dir):
      os.mkdir(self.save_dir)

    with open(os.path.join(self.save_dir, 'config_file'), 'w') as f:
      self._config.write(f)


    self._vocabs = []
    vocab_file = [(self.word_file, 'Words'),
                  (self.target_file, "Targets")]

    for i, (vocab_file, name) in enumerate(vocab_file):
      vocab = Vocab(vocab_file, self._config,
                    name = name,
                    load_embed_file = (not i),
                    lower_case = (not i)
                    )
      self._vocabs.append(vocab)

    print("################## Data ##################")
    print("There are %d words in training set" % (len(self.words) - 2))
    print("There are %d targets in training set" % (len(self.targets) - 2))
    print("Loading training set ...")
    self._trainset = Dataset(self.train_file, self._vocabs, self._config, name="Trainset")
    print("There are %d sentences in training set" % (self._trainset.sentsNum))
    print("Loading validation set ...")
    self._validset = Dataset(self.valid_file, self._vocabs, self._config, name="Validset")
    print("There are %d sentences in validation set" % (self._validset.sentsNum))
    print("Loading testing set ...")
    self._testset =  Dataset(self.test_file, self._vocabs, self._config, name="Testset")
    print("There are %d sentences in testing set" % (self._testset.sentsNum))

    self.args = {#'input_channels':2,
                 'kernel_sizes':[3,4,5],
                 'words_num': len(self.words),
                 'words_dim': self.words_dim,
                 'target_class': len(self.targets),
                 'output_channels': 100,
                 'dropout': self.dropout,
                 'embeds_num' : self.words.embeds_size,
                 'embeds_dim' : self.words_dim, # Embedding size must be the same with words size
                 'embeds':self.words.pretrained_embeddings,
                 'use_gpu': self.use_gpu,
                 'mode': self.mode}

    self.model = model
    return


  def train_minibatch(self):
    return self._trainset.minibatch(self.train_batch_size, self.input_idx, self.target_idx, shuffle=True)

  def valid_minibatch(self):
    return self._validset.minibatch(self.test_batch_size, self.input_idx, self.target_idx, shuffle=False)

  def test_minibatch(self):
    return self._testset.minibatch(self.test_batch_size, self.input_idx, self.target_idx, shuffle=False)

  def train(self):
    # if torch.cuda.is_available(): # and use_cuda
    #   self.model.cuda()
    if self.use_gpu:
      self.model = self.model(self.args).cuda()
    else:
      self.model = self.model(self.args)
    parameter = filter(lambda p: p.requires_grad, self.model.parameters())
    optimizer = torch.optim.Adam(parameter, lr=self.learning_rate)
    # The optimizer doesn't have adaptive learning rate

    step = 0
    best_score = 0
    valid_accuracy = 0
    test_accuracy = 0

    acc_corrects = 0 # count the corrects for one log_interval
    acc_sents = 0 # count sents number for one log_interval

    epoch = 0
    while True:
      for batch in self.train_minibatch():
        self.model.train()
        feature, target = batch['text'], batch['label']
        # Sanity check
        # for sent in feature:
        #   for word in sent:
        #     word_str = self.words._idx2str[word[0]]
        #     embed_str = self.words._embed2str[word[1]]
        #     if word_str != embed_str:
        #       print(word_str, embed_str)
        ##
        if self.use_gpu:
          feature = Variable(torch.from_numpy(feature).cuda())
          target = Variable(torch.from_numpy(target).cuda())[:, 0]
        else:
          feature = Variable(torch.from_numpy(feature))
          target = Variable(torch.from_numpy(target))[:, 0]

        # if torch.cuda.is_available():
        #   feature, target = feature.cuda(), target.cuda()
        optimizer.zero_grad() # Clears the gradients of all optimized Variable
        logit = self.model(feature)
        loss = F.cross_entropy(logit, target)
        loss.backward()
        optimizer.step()
        step += 1
        preds = torch.max(logit, 1)[1].view(target.size())  # get the index
        acc_corrects += (preds.cpu().data == target.cpu().data).sum()
        acc_sents += batch['batch_size']
        # if step % self.log_interval == 0:
        #   accuracy = float(acc_corrects) / float(acc_sents) * 100.0
        #   print("## [Batch %d] Accuracy : %5.2f" % (step, accuracy))
        #   acc_corrects = 0
        #   acc_sents = 0

        if step == 1 or step % self.valid_interval == 0:
          accuracy = self.test(validate=True)
          print("## Validation: %5.2f" % (accuracy))
          if accuracy > best_score:
            best_score = accuracy
            valid_accuracy = accuracy
            print("## Update Model ##")
            torch.save(self.model, self.save_model_file)

          print("## Currently the best validation: Accucacy %5.2f" % (valid_accuracy))

      epoch += 1
      accuracy = float(acc_corrects) / float(acc_sents) * 100
      print("[EPOCH] %d Accuracy: %5.2f" % (epoch, accuracy))
      acc_corrects = 0
      acc_sents = 0
      if (epoch % self.epoch_decay == 0):
        lr = self.learning_rate * (0.75 ** (epoch // self.epoch_decay))
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr


  def test(self, validate=False):
    self.model.eval()
    if validate:
      dataset = self._validset
      minibatch = self.valid_minibatch
    else:
      dataset = self._testset
      minibatch = self.test_minibatch

    test_corrects = 0
    test_sents = 0
    for batch in minibatch():
      # TODO: Prediton to Text
      feature, target = batch['text'], batch['label']
      if self.use_gpu:
        feature = Variable(torch.from_numpy(feature).cuda())
      else:
        feature = Variable(torch.from_numpy(feature))
      target = Variable(torch.from_numpy(target))[:,0]
      # if torch.cuda.is_available():
      #   feature, target = feature.cuda(), target.cuda()

      logit = self.model(feature)
      preds = torch.max(logit, 1)[1].view(target.size())  # get the index
      test_corrects += (preds.cpu().data == target.data).sum()
      test_sents += batch['batch_size']
    return float(test_corrects) / float(test_sents) * 100.0

  @property
  def words(self):
    return self._vocabs[0]

  @property
  def targets(self):
    return self._vocabs[1]


  @property
  def input_idx(self):
    return (0, 1)


  @property
  def target_idx(self):
    return (0,)



