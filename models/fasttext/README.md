## Bag of Tricks for Efficient Text Classification

Implementation of [FastText (2016)](https://arxiv.org/pdf/1607.01759.pdf)

## Quick Start

To run the model on Reuters dataset, just run the following from the Castor working directory:

```
python -m models.fasttext --dataset Reuters --batch-size 128 --lr 0.01 --seed 3435 --epochs 30
```


## Settings

Adam is used for training.
