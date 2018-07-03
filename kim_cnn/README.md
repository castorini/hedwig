# kim_cnn

Implementation for Convolutional Neural Networks for Sentence Classification of [Kim (2014)](https://arxiv.org/abs/1408.5882) with PyTorch and Torchtext.

## Model Type

- rand: All words are randomly initialized and then modified during training.
- static: A model with pre-trained vectors from [word2vec](https://code.google.com/archive/p/word2vec/). All words -- including the unknown ones that are initialized with zero -- are kept static and only the other parameters of the model are learned.
- non-static: Same as above but the pretrained vectors are fine-tuned for each task.
- multichannel: A model with two sets of word vectors. Each set of vectors is treated as a 'channel' and each filter is applied to both channels, but gradients are back-propagated only through one of the channels. Hence the model is able to fine-tune one set of vectors while keeping the other static. Both channels are initialized with word2vec.# text-classification-cnn
Implementation for Convolutional Neural Networks for Sentence Classification of [Kim (2014)](https://arxiv.org/abs/1408.5882) with PyTorch.

## Quick Start

To run the model on SST-1 dataset on multichannel, just run the following from the Castor working directory.

```
python -m kim_cnn --mode multichannel
```

The file will be saved in

```
kim_cnn/saves/best_model.pt
```

To test the model, you can use the following command.

```
python -m kim_cnn --trained_model kim_cnn/saves/SST-1/multichannel_best_model.pt --mode multichannel
```

## Dataset

We experiment the model on the following datasets.

- SST-1: Keep the original splits and train with phrase level dataset and test on sentence level dataset.

## Settings

Adadelta is used for training.

## Training Time

For training time, when

```
torch.backends.cudnn.deterministic = True
```

is specified, the training will be ~3h because deterministic cnn algorithm is used (accuracy v.s. speed).

Other option is that

```
torch.backends.cudnn.enabled = False
```
but this will take ~6-7x training time.

## SST-1 Dataset Results

**Random**

```
python -m kim_cnn --mode rand --lr 0.8337 --weight_decay 0.0008987 --dropout 0.4
```

**Static**

```
python -m kim_cnn --mode static --lr 0.8641 --weight_decay 1.44e-05 --dropout 0.3
```

**Non-static**

```
python -m kim_cnn --mode non-static --lr 0.371 --weight_decay 1.84e-05 --dropout 0.4
```

**Multichannel**

```
python -m kim_cnn --mode multichannel --lr 0.2532 --weight_decay 3.95e-05 --dropout 0.1
```

Using deterministic algorithm for cuDNN.

| Test Accuracy on SST-1         |    rand    |    static    |    non-static  |  multichannel   |
|:------------------------------:|:----------:|:------------:|:--------------:|:---------------:|
| Paper                          |    45.0    |     45.5     |      48.0      |      47.4       |
| PyTorch using above configs    |    41.5    |     44.7     |      47.4      |      47.5       |

## TODO

- More experiments on SST-2 and subjectivity
- Parameters tuning
