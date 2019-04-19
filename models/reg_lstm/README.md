# RegLSTM

Implementation of a standard LSTM with regularization using PyTorch  for text classification.

## Quick start

To run the model on Reuters dataset on static, just run the following from the project working directory.

```
python -m models.reg_lstm --dataset Reuters --mode static --batch-size 32 --lr 0.01 --epochs 30 --bidirectional --num-layers 1 --hidden-dim 512 --wdrop 0.1 --embed-droprate 0.2 --dropout 0.5 --beta-ema 0.99 --seed 3435
```

The best model weights will be saved in

```
models/reg_lstm/saves/Reuters/best_model.pt
```

To test the model, you can use the following command.

```
python -m models.reg_lstm --dataset Reuters --mode static --batch-size 32 --trained-model models/reg_lstm/saves/Reuters/best_model.pt --seed 3435
```

## Model Types

- rand: All words are randomly initialized and then modified during training.
- static: A model with pre-trained vectors from [word2vec](https://code.google.com/archive/p/word2vec/). 
    All words, including the unknown ones that are initialized with zero, are kept static and only the other 
    parameters of the model are learned.
- non-static: Same as above but the pretrained vectors are fine-tuned for each task.

## Regularization

Regularization options like embedding dropout, weight drop, temporal activation regularization and temporal averaging are available 
through command line args.

## Dataset

We experiment the model on the following datasets:

- Reuters (ModApte)
- AAPD
- IMDB
- Yelp 2014

## Settings

Adam is used for training with an option for temporal averaging.

## Acknowledgement
- The additional modules have been heavily inspired by two open source repositories:
	- https://github.com/salesforce/awd-lstm-lm.git
	- https://github.com/AMLab-Amsterdam/L0_regularization.git
