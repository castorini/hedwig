# text-classification-cnn
Implementation for Convolutional Neural Networks for Sentence Classification of [Kim (2014)](https://arxiv.org/abs/1408.5882) with PyTorch.

## Project Structure

```
text-classification-cnn
  ├── config
  │   └── classification.cfg
  ├── etc
  │   ├── kmeans.py
  │   └── utils.py
  ├── model
  │   └── cnnText
  │       └── cnntext.py
  ├── network
  │   └── cnnTextNetwork.py
  ├── data
  ├── saves
  ├── README.md
  ├── getData.sh
  ├── bucket.py
  ├── configurable.py
  ├── dataset.py
  ├── example.py
  ├── vocab.py
  └── main.py

```

## Model Type

- rand: All words are randomly initialized and then modified during training.
- static: A model with pre-trained vectors from [word2vec](https://code.google.com/archive/p/word2vec/). All words -- including the unknown ones that are randomly initialized -- are kept static and only the other parameters of the model are learned.
- non-static: Same as above but the pretrained vectors are fine-tuned for each task.
- multichannel: A model with two sets of word vectors. Each set of vectors is treated as a 'channel' and each filter is applied to both channels, but gradients are back-propagated only through one of the channels. Hence the model is able to fine-tune one set of vectors while keeping the other static. Both channels are initialized with word2vec.



## Quick Start

Run 

```
bash getData.sh
```

to get dataset.

To run the model on [TREC](http://cogcomp.cs.illinois.edu/Data/QA/QC/) dataset on [rand](Model Type), just run the following code.

```
python main.py --config_file config/trec.cfg --model_type CNNText --train
```

The file will be saved in 

```
saves/model_file
```
You can modify these parameters under these [instructions](Configurable File)

To test the model, you can use the following command.

```
python main.py --config_file config/trec.cfg --model_type CNNText --test --restore_from saves/trec/model_file
```

You need to specify the config file and the path to model file here. Note: The path need to be the same as the declaration in the config file.

## Configurable File

- model_type: **CNNText** in this case.
- mode: **rand**, **static**, **non-static** and **multichannel** which are specified [here](Model Type)
- save_dir: the path you want to save the model parameters
- word_file: all words appearing in the dataset
- target_file: all labels appearing in the dataset
- data_dir: the path of dataset
- train_file: the name of the training file
- valid_file: the name of the validation file
- test_file: the name of the test file
- save_model_file: the name you want to use for the model parameters
- restore_from: this option will be used we you want to restore file for validation and testing
- embed_file: embedding file
- use_gpu: use GPU or not
- words_dim: the dimension of the words. This is same with pre-trained word embedding.
- n_bkts: the bucket number for training dataset. Group the sentence according to the lengths
- n_valid_bkts: the bucket number for testing dataset.
- dataset_type: the dataset you used. **TREC**, **SST-1** and **SST-2** in this case.
- min_occur_count: set the word as *UNK* according to its frequency.
- learning_rate: leanring rate
- epoch_decay: decay the learning rate 0.75 for every *epoch_dacay* epoch
- valid_interval: validate for every *valid_interval* iterations
- train_batch_size: the token number for each batch in training
- test_batch_size: the token number for  each batch in testing


## Dataset and Embeddings 

We experiment the model on the following three datasets.

- TREC: We use the 1-5000 for training, 5001-5452 for validation, and original test dataset for testing.
- SST-1: Keep the original splits and train with phrase level dataset and test on sentence level dataset.
- SST-2: Same as above.

Furthermore, we filter the word embeddings to fit specific dataset. These file can be found in dir *data*.

For self-defined data, please keep the format as

```
label1 sentence
label2 sentence
```

And you can use [this](http://ocp59jkku.bkt.clouddn.com/filterVec.py) script to filter the pre-trained word embeddings.

Or you can modify the *reading_dataset* in *dataset.py*, *add_train_file* in *vocab.py* and *example.py* to fit your own dataset.


## Results


|dataset|rand|static|non-static|multichannel|
|---|---|---|---|---|
|TREC|91.98|90.32|92.62|93.36|
|SST-1|42.59|46.33|44.32|47.32|
|SST-2|82.20|86.42|85.43|84.39|

We do not tune the parameters for each dataset. And the implementation is simplified from the original version on regularization.

