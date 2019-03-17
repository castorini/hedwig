# XML_CNN

Implementation for XML Convolutional Neural Network for Document Classification of [XML-CNN (2014)](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf) with PyTorch and Torchtext.

## Model Type

- rand: All words are randomly initialized and then modified during training.
- static: A model with pre-trained vectors from [word2vec](https://code.google.com/archive/p/word2vec/). All words -- including the unknown ones that are initialized with zero -- are kept static and only the other parameters of the model are learned.
- non-static: Same as above but the pretrained vectors are fine-tuned for each task.

## Quick Start

To run the model on Reuters dataset on static just run the following from the Castor working directory.

```
python -m xml_cnn --dataset Reuters
```

The file will be saved in

```
xml_cnn/saves/best_model.pt
```



## Dataset

We experiment the model on the following datasets.

- Reuters: A multi-label document classification dataset. 

## Settings

Adam is used for training.


## TODO

- Report hyperparameters and results after finetuning on other datasets like AAPD.
