## SM model

#### References:
1. Aliaksei _S_everyn and Alessandro _M_oschitti. 2015. Learning to Rank Short Text Pairs with Convolutional Deep Neural 
Networks. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information 
Retrieval (SIGIR '15). ACM, New York, NY, USA, 373-382. DOI: http://dx.doi.org/10.1145/2766462.2767738

Please ensure you have followed instructions in the main [README](../README.md) doc before running any further commands in this doc.

Your repository root should be in your `PYTHONPATH` environment variable:
```bash
export PYTHONPATH=$(pwd)
```

To create the dataset:
```bash
cd Castor/sm_cnn/
./create_dataset.sh
```

We use `trec_eval` for evaluation:

```bash
cd ../utils/
./get_trec_eval.sh
cd ../sm_cnn
```

### Training
Download the word2vec model from [here](https://drive.google.com/file/d/0B2u_nClt6NbzUmhOZU55eEo4QWM/view?usp=sharing) 
and copy it to the `data/` folder.

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
  
### TrecQA:

#### Best dev 
Metric |rand  |static|non-static|multichannel
-------|------|------|----------|------------
MAP    |0.8096|0.8162|0.8387    | 0.8274     
MRR    |0.8560|0.8918|0.9058    | 0.8818
 
#### Test
Metric |rand   |static|non-static|multichannel
-------|-------|------|----------|------------
MAP    |0.7441 |0.7524|0.7688    |0.7641
MRR    |0.8172 |0.8012|0.8144    |0.8174

### WikiQA:

#### Best dev 
Metric |rand  |static|non-static|multichannel
-------|------|------|----------|------------
MAP    |0.7109|0.7204|0.7049    | 0.7245     
MRR    |0.7169|0.7234|0.7075    | 0.7259
 
#### Test
Metric |rand   |static|non-static|multichannel
-------|-------|------|----------|------------
MAP    |0.6313 |0.6378|0.6455    |0.6476
MRR    |0.6522 |0.6542|0.6689    |0.6646

NB: The results on WikiQA are based on the SM model hyperparameters.  


### To create your own word2vec.pt file

+ Download word2vec from [here](https://drive.google.com/drive/u/0/folders/0B-yipfgecoSBfkZlY2FFWEpDR3M4Qkw5U055MWJrenE5MTBFVXlpRnd0QjZaMDQxejh1cWs)
to the `data/` folder

```bash
python $PYTHONPATH/utils/build_w2v.py --input ../../Castor-data/embeddings/word2vec/aquaint+wiki.txt.gz.ndim=50.bin
```

Note that `$PYTHONPATH` holds the location of the repository root.
