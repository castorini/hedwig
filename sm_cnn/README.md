## SM model

#### References:
1. Aliaksei _S_everyn and Alessandro _M_oschitti. 2015. Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '15). ACM, New York, NY, USA, 373-382. DOI: http://dx.doi.org/10.1145/2766462.2767738


### Requirements
gensim==1.0.1
nltk==3.2.2
numpy==1.11.3
pandas==0.19.2
torch==0.1.11+b13b701

Please install the requirements. See [Castor/README.md](../README.md) for pytorch installation details.

### Training the model

1. Setup repository layout.

```
mkdir castorini
cd castorini
git clone https://github.com/castorini/data.git
git clone https://github.com/castorini/models.git
git clone https://github.com/castorini/Castor.git
```

This should generate:
```
├── Castor
│   ├── README.md
│   ├── idf_baseline
│   ├── kim_cnn
│   ├── simple_qa_rnn
│   └── sm_cnn/
├── data
│   ├── README.md
│   ├── TrecQA/
│   └── word2vec/
└── models
    ├── README.md
    └── sm_cnn/
```

2. Preprocess data

```
cd data/TrecQA
python3 parse.py
python3 overlap_features.py
python3 build_vocab.py
```

3. Download word embeddings from [here](https://drive.google.com/folderview?id=0B-yipfgecoSBfkZlY2FFWEpDR3M4Qkw5U055MWJrenE5MTBFVXlpRnd0QjZaMDQxejh1cWs&usp=sharing) and save the ``aquaint+wiki.txt.gz.ndim=50.bin`` into ``data/word2vec/``.


4. Train the model

Make trec_eval

```
cd Castor/sm_cnn/
cd trec_eval-8.0
make clean && make
cd ..
```

To train the S&M model on TrecQA
```
python main.py ../../models/sm_model/sm_model.TrecQA.TRAIN-ALL.2017-04-02.castor --paper-ext-feats
```
**To use the GPU, add `--cuda`.**

The final model will be saved to ```../../models/sm_model/sm_model.TrecQA.TRAIN-ALL.2017-04-02.castor```

_NOTE:_ On first run, the program will create a memory-mapped cache for word e  mbeddings (943MB) in ``data/word2vec``.
The cache allows for faster loading of data in future runs.

Run ```python main.py -h``` for more default options.

