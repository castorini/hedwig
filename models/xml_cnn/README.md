# XML_CNN

Implementation of XML Convolutional Neural Network for Document Classification [XML-CNN (2014)](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf) with PyTorch and Torchtext.

## Quick Start

To run the model on the Reuters dataset, just run the following from the working directory:

```
python -m models.xml_cnn --mode static --dataset Reuters --batch-size 32 --lr 0.01 --epochs 30 --dropout 0.5 --dynamic-pool-length 8 --seed 3435
```

The best model weight will be saved in

```
models/xml_cnn/saves/Reuters/best_model.pt
```

To test the model, you can use the following command.

```
python -m models.xml_cnn --dataset Reuters --mode static --batch-size 32 --dynamic-pool-length 8 --trained-model models/xml_cnn/saves/Reuters/best_model.pt --seed 3435
```

## Model Types

- rand: All words are randomly initialized and then modified during training.
- static: A model with pre-trained vectors from [word2vec](https://code.google.com/archive/p/word2vec/). All words -- including the unknown ones that are initialized with zero -- are kept static and only the other parameters of the model are learned.
- non-static: Same as above but the pretrained vectors are fine-tuned for each task.

## Dataset

We experiment the model on the following datasets.

- Reuters (ModApte)
- AAPD

## Settings

Adam is used for training.
