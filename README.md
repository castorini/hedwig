# Castor

This is the common repo for PyTorch deep learning models by the Data Systems Group at the University of Waterloo.

## Models

### Predictions Over One Input Sequence

For sentiment analysis, topic classification, etc.

+ [Kim CNN](./kim_cnn/): baseline convolutional neural network for sentence classification [(Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181)
+ [conv-RNN](./conv_rnn): convolutional RNN [(Wang et al., KDD 2017)](https://dl.acm.org/citation.cfm?id=3098140)

### Predictions Over Two Input Sequences

For paraphrase detection, question answering, etc.

+ [SM-CNN](./sm_cnn/): Siamese CNN for ranking texts [(Severyn and Moschitti, SIGIR 2015)](https://dl.acm.org/citation.cfm?id=2767738)
+ [MP-CNN](./mp_cnn/): Multi-Perspective CNN [(He et al., EMNLP 2015)](http://anthology.aclweb.org/D/D15/D15-1181.pdf)
+ [NCE](./nce/): Noise-Contrastive Estimation for answer selection applied on SM-CNN and MP-CNN
+ [IDF Baseline](./idf_baseline/): IDF overlap between question and candidate answers

## Setting up PyTorch

Copy and run the command at https://pytorch.org/ for your environment. PyTorch recommends the Anaconda environment, which we use in our lab.

The typical installation command is

```bash
conda install pytorch torchvision -c pytorch
```

## Data and Pre-Trained Models

Data associated for use with this repository can be found at: https://git.uwaterloo.ca/jimmylin/Castor-data.git.

Pre-trained models can be found at: https://github.com/castorini/models.git.

Your directory structure should look like
```
.
├── Castor
├── Castor-data
└── models
```

For example (if you use HTTPS instead of SSH):

```bash
git clone https://github.com/castorini/Castor.git
git clone https://git.uwaterloo.ca/jimmylin/Castor-data.git
git clone https://github.com/castorini/models.git
```

Sourcing and pre-processing of input data for each model is described in the respective ```model/README.md```'s.
