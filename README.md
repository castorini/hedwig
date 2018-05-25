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
+ [NCE](./nce/): Noise-Contrastive Estimation for answer selection applied on SM-CNN and MP-CNN [(Rao et al., CIKM 2016)](https://dl.acm.org/citation.cfm?id=2983872)
+ [IDF Baseline](./idf_baseline/): IDF overlap between question and candidate answers

Each model directory has a `README.md` with further details.

## Setting up PyTorch

**If you are an internal Castor contributor and is planning to use the Data System Group's GPU machines in the lab,
please follow the instructions [here](./docs/internal-instructions.md) instead.**

Copy and run the command at [https://pytorch.org/](https://pytorch.org/) for your environment.
PyTorch recommends the Anaconda environment, which we use in our lab. We are currently targeting PyTorch 0.4 for our codebase.

The typical installation command is

```bash
conda install pytorch torchvision -c pytorch
```

Other Python packages we use can be installed via pip:

```bash
pip install -r requirements.txt
```

Please also run the following inside the `utils` directory to build the `trec_eval` tool for evaluating certain datasets.

```bash
./get_trec_eval.sh
```

## Data and Pre-Trained Models

**If you are an internal Castor contributor and is planning to use the Data System Group's GPU machines in the lab,
please follow the instructions [here](./docs/internal-instructions.md) instead.**

Data associated for use with this repository can be found at: https://git.uwaterloo.ca/jimmylin/Castor-data.git.

Pre-trained models can be found at: https://git.uwaterloo.ca/jimmylin/Castor-models.

Your directory structure should look like
```
.
├── Castor
├── Castor-data
└── Castor-models
```

For example (if you use HTTPS instead of SSH):

```bash
git clone https://github.com/castorini/Castor.git
git clone https://git.uwaterloo.ca/jimmylin/Castor-data.git
git clone https://git.uwaterloo.ca/jimmylin/Castor-models.git
```

After cloning the Castor-data repo, you need to unzip embeddings and run data pre-processing scripts. You can choose
to follow instructions under each dataset / embedding directory separately, or just run the following script in Castor-data
to do all of the steps for you:

```bash
./setup.sh
```
