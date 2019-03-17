# LSTM with Regularization

Implementation of a standard LSTM using PyTorch and Torchtext for text classification with Regularization.

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

Adam is used for training with an option of temporal averaging.

## TODO
- Support ONNX export. Currently throws a ONNX export failed (Couldn't export Python operator forward_flattened_wrapper) exception.
- Add dataset results with different hyperparameters
- Parameters tuning

## Regularization Module

- Regularization methods like Embedding dropout, Weight Dropped LSTM and Temporal Activation Regularization are implemented.
- Temporal Averaging is also an additional module

## Acknowledgement
- The additional modules have been heavily inspired by two open source repositories:
	- https://github.com/salesforce/awd-lstm-lm.git
	- https://github.com/AMLab-Amsterdam/L0_regularization.git
