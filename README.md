# Castor

This is the common repo for PyTorch deep learning models by the Data Systems Group at the University of Waterloo.

## Models

### Predictions Over One Input Sequence

For sentiment analysis, topic classification, etc.

+ [Kim CNN](./kim_cnn/): baseline convolutional neural networks
+ [conv-RNN](./conv_rnn): convolutional RNN

### Predictions Over Two Input Sequences

For paraphrase detection, question answering, etc.

+ [SM-CNN](./sm_cnn/): Ranking short text pairs with Convolutional Neural Networks
+ [MP-CNN](./mp_cnn/): Sentence pair modelling with Multi-Perspective Convolutional Neural Networks
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
