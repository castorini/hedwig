## Character-level Convolutional Neural Network

Implementation of [Char-CNN (2015)](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)

## Quick Start

To run the model on Reuters dataset, just run the following from the Castor working directory:

```
python -m models.char_cnn --dataset Reuters --batch-size 128 --lr 0.001 --seed 3435
```

The best model weights will be saved in

```
models/char_cnn/saves/Reuters/best_model.pt
```

To test the model, you can use the following command.

```
python -m models.char_cnn --dataset Reuters --batch_size 32 --trained-model models/char_cnn/saves/Reuters/best_model.pt --seed 3435
```
## Dataset

We experiment the model on the following datasets.

- Reuters (ModApte)
- AAPD

## Settings

Adam is used for training.
