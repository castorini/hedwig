# Castor

This repo contains PyTorch deep learning models for document classification, implemented by the Data Systems Group at the University of Waterloo.

## Models

+ [Kim CNN](./kim_cnn/): Baseline convolutional neural network for sentence classification [(Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181)
+ [Conv-RNN](./conv_rnn/): Convolutional RNN [(Wang et al., KDD 2017)](https://dl.acm.org/citation.cfm?id=3098140)
+ [HAN](./han/): Hierarchical Attention Networks [(Zichao, et al, NAACL 2016)](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)
+ [LSTM-Reg](./lstm_regularization/): Standard LSTM with Regularization [(Merity et al.)](https://arxiv.org/abs/1708.02182)
+ [XML-CNN](./xml_cnn/): CNNs for Extreme Multi-label Text Classification [(Liu et al., SIGIR 2017)](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf)
+ [Char-CNN](.//): Character-level Convolutional Network [(Zhang et al., NIPS 2015)](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)

Each model directory has a `README.md` with further details.

## Setting up PyTorch

**If you are an internal Hedwig contributor using GPU machines in the lab, follow the instructions [here](./docs/internal-instructions.md).**

Hedwig is designed for Python 3.6 and [PyTorch](https://pytorch.org/) 0.4.
PyTorch recommends [Anaconda](https://www.anaconda.com/distribution/) for managing your environment.
We'd recommend creating a custom environment as follows:

```
$ conda create --name castor python=3.6
$ source activate castor
```

And installing the packages as follows:

```
$ conda install pytorch torchvision -c pytorch
```

Other Python packages we use can be installed via pip:

```
$ pip install -r requirements.txt
```

Code depends on data from NLTK (e.g., stopwords) so you'll have to download them. Run the Python interpreter and type the commands:

```python
>>> import nltk
>>> nltk.download()
```

Finally, run the following inside the `utils` directory to build the `trec_eval` tool for evaluating certain datasets.

```bash
$ ./get_trec_eval.sh
```

## Data and Pre-Trained Models

**If you are an internal Hedwig contributor using GPU machines in the lab, follow the instructions [here](./docs/internal-instructions.md).**

To fully take advantage of code here, clone these other two repos:

+ [`Castor-data`](https://git.uwaterloo.ca/jimmylin/Castor-data): embeddings, datasets, etc.
+ [`Caster-models`](https://git.uwaterloo.ca/jimmylin/Castor-models): pre-trained models

Organize your directory structure as follows:

```
.
├── Hedwig
├── Castor-data
└── Castor-models
```

For example (using HTTPS):

```bash
$ git clone https://github.com/castorini/hedwig.git
$ git clone https://git.uwaterloo.ca/jimmylin/Castor-data.git
$ git clone https://git.uwaterloo.ca/jimmylin/Castor-models.git
```

After cloning the Hedwig-data repo, you need to unzip embeddings and run data pre-processing scripts. You can choose
to follow instructions under each dataset and embedding directory separately, or just run the following script in 
Hedwig-data to do all of the steps for you:

```bash
$ ./setup.sh
```
