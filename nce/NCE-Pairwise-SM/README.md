## NCE-SM model

#### References:
+ Aliaksei _S_everyn and Alessandro _M_oschitti. 2015. Learning to Rank Short Text Pairs with Convolutional Deep Neural
Networks. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information 
Retrieval (SIGIR '15). ACM, New York, NY, USA, 373-382. DOI: http://dx.doi.org/10.1145/2766462.2767738

+ Jinfeng Rao, Hua He, and Jimmy Lin. [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks.](http://dl.acm.org/citation.cfm?id=2983872) *Proceedings of the 25th ACM International on Conference on Information and Knowledge Management (CIKM 2016)*, pages 1913-1916.


The code uses torchtext for text processing. Set torchtext:
```bash
git clone https://github.com/pytorch/text.git
cd text
python setup.py install
```

Download the word2vec model from [here] (https://drive.google.com/file/d/0B2u_nClt6NbzUmhOZU55eEo4QWM/view?usp=sharing) 
and copy it to the `Castor/data/word2vec` folder.

### Training the model

You can train the SM model for the 4 following configurations:
1. __random__ - the word embedddings are initialized randomly and are tuned during training
2. __static__ - the word embeddings are static (Severyn and Moschitti, SIGIR'15)
3. __non-static__ - the word embeddings are tuned during training
4. __multichannel__ - contains static and non-static channels for question and answer conv layers


```bash
python train.py --no_cuda --mode rand --batch_size 64 --neg_num 8 --dev_every 50 --patience 1000
```

NB: pass `--no_cuda` to use CPU

The trained model will be save to:
```
saves/static_best_model.pt
```

### Testing the model

```
python main.py --trained_model saves/TREC/multichannel_best_model.pt --batch_size 64 --no_cuda
```

### Evaluation

#### The performance on TrecQA dataset:

##### Without NCE

Metric |rand   |static|non-static|multichannel
-------|-------|------|----------|------------
MAP    |0.7441 |0.7524|0.7688    |0.7641
MRR    |0.8172 |0.8012|0.8144    |0.8174

##### Max Neg Sample

To be added

##### Pairwise + Max Neg Sample with neg_num = 8

Metric |rand   |static|non-static|multichannel
-------|-------|------|----------|------------
MAP    |0.7427 |0.7546|0.7716    |0.7794
MRR    |0.8151 |0.8061|0.8347    |0.8467


#### The performance on WikiQA dataset:

To be added

