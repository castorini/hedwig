# KimCNN

Implementation for Convolutional Neural Networks for Sentence Classification of [Kim (2014)](https://arxiv.org/abs/1408.5882) with PyTorch and Torchtext.

## Quick Start

To run the model on the Reuters dataset, just run the following from the working directory:

```
python -m models.kim_cnn --mode static --dataset Reuters --batch-size 32 --lr 0.01 --epochs 30 --dropout 0.5 --seed 3435
```

The best model weights will be saved in

```
models/kim_cnn/saves/Reuters/best_model.pt
```

To test the model, you can use the following command.

```
python -m models.kim_cnn --dataset Reuters --mode static --batch-size 32 --trained-model models/kim_cnn/saves/Reuters/best_model.pt --seed 3435
```

## Model Types

- rand: All words are randomly initialized and then modified during training.
- static: A model with pre-trained vectors from [word2vec](https://code.google.com/archive/p/word2vec/). 
    All words, including the unknown ones that are initialized with zero, are kept static and only the other 
    parameters of the model are learned.
- non-static: Same as above but the pretrained vectors are fine-tuned for each task.
- multichannel: A model with two sets of word vectors. Each set of vectors is treated as a 'channel' and each 
    filter is applied to both channels, but gradients are back-propagated only through one of the channels. Hence the 
    model is able to fine-tune one set of vectors while keeping the other static. Both channels are initialized with 
    word2vec.

## Dataset

We experiment the model on the following datasets:

- Reuters (ModApte)
- AAPD
- IMDB
- Yelp 2014

## Settings

Adam is used for training.
