from setuptools import setup

setup(name='hedwig',
      version='1.0.0',
      description='PyTorch deep learning models for document classification',
      packages=['models/char_cnn', 'models/han', 'models/kim_cnn', 'models/reg_lstm', 'models/xml_cnn'],
      )
