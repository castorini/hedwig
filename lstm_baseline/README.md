# lstm_baseline

Implementation of a standard LSTM using PyTorch and Torchtext for text classification baseline measurements.

## Model Type

- rand: All words are randomly initialized and then modified during training.
- static: A model with pre-trained vectors from [word2vec](https://code.google.com/archive/p/word2vec/). All words -- including the unknown ones that are initialized with zero -- are kept static and only the other parameters of the model are learned.
- non-static: Same as above but the pretrained vectors are fine-tuned for each task.

## Quick Start

To run the model on Reuters dataset on static, just run the following from the Castor working directory.

```
python -m lstm_baseline --mode static
```

## Dataset

We experiment the model on the following datasets.

- Reuters dataset - ModApte splits

## Settings

Adadelta is used for training.

## TODO
- Support ONNX export. Currently throws a ONNX export failed (Couldn't export Python operator forward_flattened_wrapper) exception.
- Add dataset results with different hyperparameters
- Parameters tuning
