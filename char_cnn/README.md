## Character-level Convolutional Network

Implementation of Char-CNN from Character-level Convolutional Networks for Text Classification (http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)

## Quick Start

To run the model on Reuters dataset, just run the following from the Castor working directory:

```
python -m char_cnn --dataset Reuters --gpu 1 --batch_size 128 --lr 0.001
```

To test the model, you can use the following command.

```
python -m char_cnn --trained_model kim_cnn/saves/Reuters/best_model.pt
```

## Dataset

We experiment the model on the following datasets.

- Reuters Newswire (RCV-1)
- Arxiv Academic Paper Dataset (AAPD)

## Settings

Adam is used for training.

## Dataset Results

### RCV-1
```
python -m char_cnn --dataset Reuters --gpu 1 --batch_size 128 --lr 0.001
```
  | Accuracy | Avg. Precision | Avg. Recall | Avg. F1
-- | -- | -- | -- | --
Char-CNN (Dev) | 0.585 | 0.702 | 0.569 | 0.628
Char-CNN (Test) | 0.589 | 0.691 | 0.552 | 0.614

### AAPD
```
python -m char_cnn --dataset AAPD --gpu 1 --batch_size 128 --lr 0.001
```
  | Accuracy | Avg. Precision | Avg. Recall | Avg. F1
-- | -- | -- | -- | --
Char-CNN (Dev) | 0.305 | 0.681 | 0.537 | 0.600
Char-CNN (Test) | 0.294 | 0.681 | 0.526 | 0.593

## TODO
- Support ONNX export. Currently throws a ONNX export failed (Couldn't export Python operator forward_flattened_wrapper) exception.
- Parameters tuning

