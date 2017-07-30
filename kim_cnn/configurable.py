#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

from configparser import SafeConfigParser


class Configurable(object):
  """
  Configuration processing for the network
  """
  def __init__(self, *args, **kwargs):
    self._name = kwargs.pop("name", "Unknown")
    if args and kwargs:
      raise TypeError('Configurable must take either a config parser or keyword args')
    if len(args) > 1:
      raise TypeError('Configurable takes at most one argument')
    if args:
      self._config = args[0]
    else:
      self._config = self._configure(**kwargs)
    return

  @property
  def name(self):
    return self._name


  def _configure(self, **kwargs):
    config = SafeConfigParser()
    config_file = kwargs.pop("config_file", "")
    config.read(config_file)
    # Override the config setting if the (k,v) specified in command line
    for option, value in kwargs.items():
      assigned = False
      for section in config.sections():
        if option in config.options(section):
          config.set(section, option, str(value))
          assigned = True
          break
      if not assigned:
        raise ValueError("%s is not a valid option" % option)
    return config

  argparser = argparse.ArgumentParser()
  argparser.add_argument('--config_file')

  # ======
  # [OS]
  @property
  def model_type(self):
    return self._config.get('OS', 'model_type')
  argparser.add_argument('--model_type')
  @property
  def mode(self):
    return self._config.get('OS', 'mode')
  argparser.add_argument('--mode')
  @property
  def save_dir(self):
    return self._config.get('OS', 'save_dir')
  argparser.add_argument('--save_dir')
  @property
  def word_file(self):
    return self._config.get('OS', 'word_file')
  argparser.add_argument('--word_file')
  @property
  def target_file(self):
    return self._config.get('OS', 'target_file')
  argparser.add_argument('--target_file')
  @property
  def train_file(self):
    return self._config.get('OS', 'train_file')
  argparser.add_argument('--train_file')
  @property
  def valid_file(self):
    return self._config.get('OS', 'valid_file')
  argparser.add_argument('--valid_file')
  @property
  def test_file(self):
    return self._config.get('OS', 'test_file')
  argparser.add_argument('--test_file')
  @property
  def save_model_file(self):
    return self._config.get('OS', 'save_model_file')
  argparser.add_argument('--save_model_file')
  @property
  def restore_from(self):
    return self._config.get('OS', 'restore_from')
  argparser.add_argument('--restore_from')
  @property
  def embed_file(self):
    return self._config.get('OS', 'embed_file')
  argparser.add_argument('--embed_file')
  @property
  def use_gpu(self):
    return self._config.getboolean('OS', 'use_gpu')
  argparser.add_argument('--use_gpu')


  # [Dataset]
  @property
  def n_bkts(self):
    return self._config.getint('Dataset', 'n_bkts')
  argparser.add_argument('--n_bkts')
  @property
  def n_valid_bkts(self):
    return self._config.getint('Dataset', 'n_valid_bkts')
  argparser.add_argument('--n_valid_bkts')
  @property
  def dataset_type(self):
    return self._config.get('Dataset', 'dataset_type')
  argparser.add_argument('--dataset_type')
  @property
  def min_occur_count(self):
    return self._config.getint('Dataset', 'min_occur_count')
  argparser.add_argument('--min_occur_count')



  # [Learning rate]
  @property
  def learning_rate(self):
    return self._config.getfloat('Learning rate', 'learning_rate')
  argparser.add_argument('--learning_rate')
  @property
  def epoch_decay(self):
    return self._config.getint('Learning rate', 'epoch_decay')
  argparser.add_argument('--epoch_decay')
  @property
  def dropout(self):
    return self._config.getfloat('Learning rate', 'dropout')
  argparser.add_argument('--dropout')

  # [Sizes]
  @property
  def words_dim(self):
    return self._config.getint('Sizes', 'words_dim')
  argparser.add_argument('--words_dim')


  # [Training]
  @property
  def log_interval(self):
    return self._config.getint('Training', 'log_interval')
  argparser.add_argument('--log_interval')
  @property
  def valid_interval(self):
    return self._config.getint('Training', 'valid_interval')
  argparser.add_argument('--valid_interval')
  @property
  def train_batch_size(self):
    return self._config.getint('Training', 'train_batch_size')
  argparser.add_argument('--train_batch_size')
  @property
  def test_batch_size(self):
    return self._config.getint('Training', 'test_batch_size')
  argparser.add_argument('--test_batch_size')



