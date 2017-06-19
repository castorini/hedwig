Directions:
1. Download SimpleQuestions data from this [link](https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz) and put it in directory "data/SimpleQuestions_v2/"
2. Run this script to download GloVe word embeddings and do some preprocessing.
```
bash fetch_and_preprocess.sh
```
3. Run this command to train the model. Make sure you have PyTorch and other Python dependencies installed.
```
python train_relation_model.py 
```
For GPU, use:
```
python train_relation_model.py --cuda
```