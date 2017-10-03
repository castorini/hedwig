## SM model

#### References:
1. Aliaksei _S_everyn and Alessandro _M_oschitti. 2015. Learning to Rank Short Text Pairs with Convolutional Deep Neural 
Networks. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information 
Retrieval (SIGIR '15). ACM, New York, NY, USA, 373-382. DOI: http://dx.doi.org/10.1145/2766462.2767738


### Requirements
```
nltk==3.2.2
numpy==1.11.3
pytorch==0.1.12
```

The code uses torchtext for text processing. Set torchtext:
```bash
git clone https://github.com/pytorch/text.git
cd text
python setup.py install
```

We use `trec_eval` for evaluation:

```bash
cd eval
tar -xvf trec_eval.9.0.tar.gz
make
cd ..
```

Download the word2vec model from [here] (https://drive.google.com/file/d/0B2u_nClt6NbzUmhOZU55eEo4QWM/view?usp=sharing) 
and copy it to the `data/` folder.

### Training the model

You can train the SM model for the 4 following configurations:
1. __random__ - the word embedddings are initialized randomly and are tuned during training
2. __static__ - the word embeddings are static (Severyn and Moschitti, SIGIR'15)
3. __non-static__ - the word embeddings are tuned during training
4. __multichannel__ - contains static and non-static channels for question and answer conv layers

To train on GPU 0 with static configuration:

```bash
python train.py --mode static --gpu 0
```

NB: pass `--no_cuda` to use CPU

The trained model will be save to:
```
saves/static_best_model.pt
```

### Testing the model

```
python main.py --trained_model saves/TREC/multichannel_best_model.pt 
```

### Evaluation

The performance on TrecQA dataset:
  
### Best dev 

Metric |rand  |static|non-static|multichannel
-------|------|------|----------|------------
MAP    |0.8096|0.8162|0.8387    | 0.8274     
MRR    |0.8560|0.8918|0.9058    | 0.8818
 
### Test

Metric |rand   |static|non-static|multichannel
-------|-------|------|----------|------------
MAP    |0.7441 |0.7524|0.7688    |0.7641
MRR    |0.8172 |0.8012|0.8144    |0.8174