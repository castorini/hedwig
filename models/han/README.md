# Hierarchical Attention Networks

Implementation for Hierarchical Attention Networks for Documnet Classification of [HAN (2016)](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf) with PyTorch and Torchtext.

## Model Type

- rand: All words are randomly initialized and then modified during training.
- static: A model with pre-trained vectors from [word2vec](https://code.google.com/archive/p/word2vec/). All words -- including the unknown ones that are initialized with zero -- are kept static and only the other parameters of the model are learned.
- non-static: Same as above but the pretrained vectors are fine-tuned for each task.



## Quick Start

To run the model on Reuters dataset on static, just run the following from the Castor working directory.

```
python -m han --dataset Reuters 
```

The file will be saved in

```
han/saves/best_model.pt
```

To test the model, you can use the following command.

```
python -m han --trained_model han/saves/Reuters/static_best_model.pt 
```

## Dataset

We experiment the model on the following datasets.

- Reuters-21578: Split the data into sentences for the sentence level attention model and split the sentences into words for the word level attention. The word2vec pretrained embeddings were used for the task.

## Settings

Adam is used for training.

## Training Time

For training time, when

```
torch.backends.cudnn.deterministic = True
```

is specified, the training will be ~10 min. Reuters-21578 is a relatively small dataset and the implementation is a vectorized one, hence the speed. 



## TODO
- a combined hyperparameter tuning on a few of the datasets and report results with the hyperparameters
