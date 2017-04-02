## SM model 

#### References:
1. Aliaksei _S_everyn and Alessandro _M_oschitti. 2015. Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '15). ACM, New York, NY, USA, 373-382. DOI: http://dx.doi.org/10.1145/2766462.2767738


#### TODOs:
1. figure out if the L2 regularization is correct
2. Batch size of 50 (current batch_size = 1)

#### Requirements
gensim==1.0.1
nltk==3.2.2
numpy==1.11.3
pandas==0.19.2
torch==0.1.11+b13b701

#### Getting the data

git clone [castorini/data](https://github.com/castorini/data)

castorini/data contains:

```word2vec/aquaint+wiki.txt.gz.ndim=50.bin```: word embeddings.
Note that a memory mapped cache will be created on first use on your disk, when you run ```main.py``` (below).

```TrecQA/```: the directory with the input data for training the model.

Follow instructions in castorini/data/TrecQA/README.md to preprocess data for it to be ingestable by the model.


#### Running the model

``1.`` Make TrecEval:
```
$ cd trec_eval-8.0
$ make clean
$ make
```

``2.`` To run the S&M model on TrecQA, please follow the same parameter setting:
```
$ python main.py  ../../model/sm.model.aquaint.train-all --train_all
```
The final model will be saved to ```../../model/sm.model.aquaint.train-all```

Run ```python main.py -h``` for more default options.

