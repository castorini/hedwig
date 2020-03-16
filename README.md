<p align="center">
<img src="https://github.com/karkaroff/hedwig/blob/bellatrix/docs/hedwig.png" width="360">
</p>

This repo contains PyTorch deep learning models for document classification, implemented by the Data Systems Group at the University of Waterloo.

## Models

+ [DocBERT](models/bert/) : DocBERT: BERT for Document Classification [(Adhikari et al., 2019)](https://arxiv.org/abs/1904.08398v1)
+ [Reg-LSTM](models/reg_lstm/): Regularized LSTM for document classification [(Adhikari et al., NAACL 2019)](https://cs.uwaterloo.ca/~jimmylin/publications/Adhikari_etal_NAACL2019.pdf)
+ [XML-CNN](models/xml_cnn/): CNNs for extreme multi-label text classification [(Liu et al., SIGIR 2017)](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf)
+ [HAN](models/han/): Hierarchical Attention Networks [(Zichao et al., NAACL 2016)](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)
+ [Char-CNN](models/char_cnn/): Character-level Convolutional Network [(Zhang et al., NIPS 2015)](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
+ [Kim CNN](models/kim_cnn/): CNNs for sentence classification [(Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181)

Each model directory has a `README.md` with further details.

## Setting up PyTorch

Hedwig is designed for Python 3.6 and [PyTorch](https://pytorch.org/) 0.4.
PyTorch recommends [Anaconda](https://www.anaconda.com/distribution/) for managing your environment.
We'd recommend creating a custom environment as follows:

```
$ conda create --name castor python=3.6
$ source activate castor
```

And installing PyTorch as follows:

```
$ conda install pytorch=0.4.1 cuda92 -c pytorch
```

Other Python packages we use can be installed via pip:

```
$ pip install -r requirements.txt
```

Code depends on data from NLTK (e.g., stopwords) so you'll have to download them. 
Run the Python interpreter and type the commands:

```python
>>> import nltk
>>> nltk.download()
```

## Datasets

There are two ways to download the Reuters, AAPD, and IMDB datasets, along with word2vec embeddings:

Option 1. Our [Wasabi](https://wasabi.com/)-hosted mirror:

```bash
$ wget http://nlp.rocks/hedwig -O hedwig-data.zip
$ unzip hedwig-data.zip
```

Option 2. Our school-hosted repository, [`hedwig-data`](https://git.uwaterloo.ca/jimmylin/hedwig-data):

```bash
$ git clone https://github.com/castorini/hedwig.git
$ git clone https://git.uwaterloo.ca/jimmylin/hedwig-data.git
```

Next, organize your directory structure as follows:

```
.
├── hedwig
└── hedwig-data
```

After cloning the hedwig-data repo, you need to unzip the embeddings and run the preprocessing script:

```bash
cd hedwig-data/embeddings/word2vec 
gzip -d GoogleNews-vectors-negative300.bin.gz 
python bin2txt.py GoogleNews-vectors-negative300.bin GoogleNews-vectors-negative300.txt 
```

**If you are an internal Hedwig contributor using the machines in the lab, follow the instructions [here](docs/internal-instructions.md).**
