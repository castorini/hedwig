## Relation Prediction Model

- Download and extract SimpleQuestions dataset by running the script:
```
bash fetch_dataset.sh 
```

- You will also require the package - [torchtext](https://github.com/pytorch/text).
```
git clone https://github.com/pytorch/text.git
cd path/to/torchtext
python setup.py install
```

- Run the training script with the following commands. Please check out args.py file to see the different commands available:
```
cd relation_prediction
python train.py
python train.py --no_cuda
python train.py --rnn_type gru
```
