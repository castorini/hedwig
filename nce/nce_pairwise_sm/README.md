## NCE-SM-CNN model PyTorch Implementation

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
python train.py --no_cuda --mode rand --batch_size 64 --neg_num 8 --dev_every 50 --patience 100 --dataset trec
```

NB: pass `--no_cuda` to use CPU

The trained model will be save to:
```
saves/static_best_model.pt
```

### Testing the model

```
python main.py --trained_model saves/trec/multichannel_best_model.pt --batch_size 64 --no_cuda --dataset trec
```

### Evaluation

#### The performance on TrecQA dataset:

##### Without NCE

Metric |rand   |static|non-static|multichannel
-------|-------|------|----------|------------
MAP    |0.7441 |0.7524|0.7688    |0.7641
MRR    |0.8172 |0.8012|0.8144    |0.8174

##### Pairwise + Random Sample with neg_num = 8

Metric |rand   |static|non-static|multichannel
-------|-------|------|----------|------------
MAP    |0.7427 |0.7546|0.7614    | 0.7645
MRR    |0.8151 |0.8061|0.8162    | 0.8270

##### Pairwise + Max Neg Sample with neg_num = 8

Metric |rand   |static|non-static|multichannel
-------|-------|------|----------|------------
MAP    |0.7437 |0.7602|0.7752   |0.7664
MRR    |0.8151 |0.8109 |0.8270    |0.8347


#### The performance on WikiQA dataset:

##### Without NCE

Metric |rand   |static|non-static|multichannel
-------|-------|------|----------|------------
MAP    |0.6472 |0.6500 | 0.6620|0.6542
MRR    |0.664 |0.6693 | 0.6806|0.6722

##### Pairwise + Random Sample with neg_num = 8

Metric |rand   |static|non-static|multichannel
-------|-------|------|----------|------------
MAP    |0.6655 |0.6816|0.6697  |0.6739
MRR    |0.6831 |0.6992 |0.6929 |0.6925

##### Pairwise + Max Neg Sample with neg_num = 8

Metric |rand   |static|non-static|multichannel
-------|-------|------|----------|------------
MAP    |0.6687 |0.6796|0.6854   |0.6851
MRR    |0.6864 |0.6977 |0.7012   |0.7035


## Optional Dependencies

To optionally visualize the learning curve during training, we make use of https://github.com/lanpa/tensorboard-pytorch to connect to [TensorBoard](https://github.com/tensorflow/tensorboard). These projects require TensorFlow as a dependency, so you need to install TensorFlow before running the commands below. After these are installed, just add `--tensorboard` when running `train.py` and open TensorBoard in the browser.

```sh
pip install tensorboardX
pip install tensorflow-tensorboard
```

Usage:

```sh
tensorboard --host 0.0.0.0 --port 5001 --logdir runs
```