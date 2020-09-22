## Character-level Convolutional Neural Network

Implementation of [Char-CNN (2015)](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)

## Quick Start

To run the model on Reuters dataset, just run the following from the Castor working directory:

```
python -m models.char_cnn --dataset Reuters --batch-size 32 --lr 0.01 --seed 3435
```

in order to run the implementation of the paper, run the following from the Castor working directory:

```
python -m models.char_cnn --dataset Reuters --batch-size 32 --lr 0.01 --seed 3435 --using_fixed True
```

The best model weights will be saved in

```
models/char_cnn/saves/Reuters/best_model.pt
```

To test the model, you can use the following command.

```
python -m models.char_cnn --dataset Reuters --batch_size 32 --trained-model models/char_cnn/saves/Reuters/best_model.pt --seed 3435
```


**It should be noted** that the version that follows the implementation of the [Char-CNN (2015)](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf) needs to be run with the following (or similar) parameters otherwise it produces an dev F1 of 0 on Reuters:
```
python3 -m models.char_cnn --dataset Reuters --batch-size 1 --lr 0.1 --seed 3435 --using_fixed True --epochs 30 --patience 30
```

## Settings

Adam is used for training.
