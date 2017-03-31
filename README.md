# Castor

Pytorch deep learning models.

1. [SM model](./sm-model/README.md): Similarity between question and candidate answers.


## Setting up Pytorch

You need Python 3.6 to use the models in this repository.

As per [pytorch.org](pytorch.org) "Anaconda is our recommended package manager"

```conda install pytorch torchvision -c soumith```

Other pytorch installation modalities (e.g. via ```pip```) can be seen at [pytorch.org](pytorch.org).

We also recommend [gensim](https://radimrehurek.com/gensim/). We use some gensim modules to cache word embeddings.

```conda install gensim```


## Castor-data

Sourcing and pre-processing of input data for each model is described in respective ```model/README.md```'s