# Castor

PyTorch deep learning models.

1. [SM model](./sm_cnn/): Similarity between question and candidate answers.


## Setting up PyTorch

You need Python 3.6 to use the models in this repository.

As per [pytorch.org](pytorch.org),
> "[Anaconda](https://www.continuum.io/downloads) is our recommended package manager"

```conda install pytorch torchvision -c soumith```

Other pytorch installation modalities (e.g. via ```pip```) can be seen at [pytorch.org](pytorch.org).

We also recommend [gensim](https://radimrehurek.com/gensim/). We use some gensim modules to cache word embeddings.

```conda install gensim```


PyTorch has good support for GPU computations.
CUDA installation guide for linux can be found [here](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

**NOTE**: Install CUDA libraries **before** installing conda and pytorch.


## data for models

Sourcing and pre-processing of input data for each model is described in respective ```model/README.md```'s

## Baselines

1. [IDF Baseline](./idf_baseline/): IDF overlap between question and candidate answers.

## Tutorials

SM Model tutorial: [sm_cnn/tutorial.ipynb](sm_cnn/tutorial.ipynb) - notebook that walks through SM CNN model, good for beginnners.
