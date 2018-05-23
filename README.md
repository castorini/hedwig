# Castor

Deep learning for information retrieval with PyTorch.

## Models

### Baselines

1. [IDF Baseline](./idf_baseline/): IDF overlap between question and candidate answers

### Deep Learning Models

1. [SM-CNN](./sm_cnn/): Ranking short text pairs with Convolutional Neural Networks
2. [Kim CNN](./kim_cnn/): Sentence classification using Convolutional Neural Networks
3. [MP-CNN](./mp_cnn/): Sentence pair modelling with Multi-Perspective Convolutional Neural Networks
4. [NCE](./nce/): Noise-Contrastive Estimation for answer selection applied on SM-CNN and MP-CNN
5. [conv-RNN](./conv_rnn): Convolutional RNN for text modelling

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
