# Castor

This is the common repo for PyTorch deep learning models by the Data Systems Group at the University of Waterloo.

## Models

### Predictions Over One Input Text Sequence

For sentiment analysis, topic classification, etc.

+ [Kim CNN](./kim_cnn/): baseline convolutional neural network for sentence classification [(Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181)
+ [conv-RNN](./conv_rnn): convolutional RNN [(Wang et al., KDD 2017)](https://dl.acm.org/citation.cfm?id=3098140)

### Predictions Over Two Input Text Sequences

For paraphrase detection, question answering, etc.

+ [SM-CNN](./sm_cnn/): Siamese CNN for ranking texts [(Severyn and Moschitti, SIGIR 2015)](https://dl.acm.org/citation.cfm?id=2767738)
+ [MP-CNN](./mp_cnn/): Multi-Perspective CNN [(He et al., EMNLP 2015)](http://anthology.aclweb.org/D/D15/D15-1181.pdf)
+ [NCE](./nce/): Noise-Contrastive Estimation for answer selection applied on SM-CNN and MP-CNN [(Rao et al., CIKM 2016)](https://dl.acm.org/citation.cfm?id=2983872)
+ [VDPWI](./vdpwi): Very-Deep Pairwise Word Interaction NNs for modeling textual similarity [(He and Lin, NAACL 2016)](http://www.aclweb.org/anthology/N16-1108)
+ [IDF Baseline](./idf_baseline/): IDF overlap between question and candidate answers

Each model directory has a `README.md` with further details.

## Setting up PyTorch

**If you are an internal Castor contributor using GPU machines in the lab, follow the instructions [here](./docs/internal-instructions.md).**

Castor is designed for Python 3.6 and [PyTorch](https://pytorch.org/) 0.4.
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

**If you are an internal Castor contributor using GPU machines in the lab, follow the instructions [here](./docs/internal-instructions.md).**

To fully take advantage of code here, clone these other two repos:

+ [`Castor-data`](https://git.uwaterloo.ca/jimmylin/Castor-data): embeddings, datasets, etc.
+ [`Caster-models`](https://git.uwaterloo.ca/jimmylin/Castor-models): pre-trained models

Organize your directory structure as follows:

```
.
├── Castor
├── Castor-data
└── Castor-models
```

For example (using HTTPS):

```bash
$ git clone https://github.com/castorini/Castor.git
$ git clone https://git.uwaterloo.ca/jimmylin/Castor-data.git
$ git clone https://git.uwaterloo.ca/jimmylin/Castor-models.git
```

After cloning the Castor-data repo, you need to unzip embeddings and run data pre-processing scripts. You can choose
to follow instructions under each dataset and embedding directory separately, or just run the following script in Castor-data
to do all of the steps for you:

```bash
$ ./setup.sh
```
