## Similarity Measure model (SM model)

#### References:
1. Aliaksei Severyn and Alessandro Moschitti. 2015. Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '15). ACM, New York, NY, USA, 373-382. DOI: http://dx.doi.org/10.1145/2766462.2767738


#### TODOs:
1. figure out if the L2 regularization is correct
2. Batch size of 50 (current batch_size = 1)

#### Getting the data

TODO:

#### Running it

``1.`` Make TrecEval:
```
$ cd trec_eval-8.0
$ make clean
$ make
```

``2.`` Get the Overlapping features for Q and A:
```
$ python overlap_features.py ../../data/TrecQA
```

``3.`` To run the S&M model on TrecQA, please follow the same parameter setting:
```
$ python main.py ../../data/aquaint+wiki.txt.gz.ndim\=50.bin ../../data/TrecQA/ sm --batch_size 1
```

