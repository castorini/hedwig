## Bag of Tricks for Efficient Text Classification

Implementation of [FastText (2016)](https://arxiv.org/pdf/1607.01759.pdf)

## Quick Start

To run the model on Reuters dataset, just run the following from the Castor working directory:

```
python -m models.fasttext --dataset Reuters --batch-size 128 --lr 0.001 --seed 3435
```

The best model weights will be saved in

```
models/fasttext/saves/Reuters/best_model.pt
```

To test the model, you can use the following command.

```
python -m models.fasttext --dataset Reuters --batch_size 32 --trained-model models/fasttext/saves/Reuters/best_model.pt --seed 3435
```

## Settings

Adam is used for training.
