# Hierarchical Attention Networks

Implementation of Hierarchical Attention Networks for Documnet Classification [HAN (2016)](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf) with PyTorch and Torchtext.

## Quick Start

To run the model on Reuters dataset on static, just run the following from the project working directory.

```
python -m models.han --dataset Reuters --mode static --batch-size 32 --lr 0.01 --epochs 30 --seed 3435
```

The best model weights will be saved in

```
models/han/saves/Reuters/best_model.pt
```

To test the model, you can use the following command.

```
python -m models.han --dataset Reuters --mode static --batch-size 32 --trained-model models/han/saves/Reuters/best_model.pt --seed 3435
```

## Model Types

- rand: All words are randomly initialized and then modified during training.
- static: A model with pre-trained vectors from [word2vec](https://code.google.com/archive/p/word2vec/). All words -- including the unknown ones that are initialized with zero -- are kept static and only the other parameters of the model are learned.
- non-static: Same as above but the pretrained vectors are fine-tuned for each task.

## Dataset

We experiment the model on the following datasets.

- Reuters (ModApte)
- AAPD
- IMDB
- Yelp 2014

## Settings

Adam is used for training.
