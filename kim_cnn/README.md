# kim_cnn

Implementation for Convolutional Neural Networks for Sentence Classification of [Kim (2014)](https://arxiv.org/abs/1408.5882) with PyTorch and Torchtext.

## Model Type

- rand: All words are randomly initialized and then modified during training.
- static: A model with pre-trained vectors from [word2vec](https://code.google.com/archive/p/word2vec/). All words -- including the unknown ones that are initialized with zero -- are kept static and only the other parameters of the model are learned.
- non-static: Same as above but the pretrained vectors are fine-tuned for each task.
- multichannel: A model with two sets of word vectors. Each set of vectors is treated as a 'channel' and each filter is applied to both channels, but gradients are back-propagated only through one of the channels. Hence the model is able to fine-tune one set of vectors while keeping the other static. Both channels are initialized with word2vec.# text-classification-cnn
Implementation for Convolutional Neural Networks for Sentence Classification of [Kim (2014)](https://arxiv.org/abs/1408.5882) with PyTorch.


## Quick Start


To run the model on [SST-1] dataset on [multichannel](Model Type), just run the following code.

```
python train.py --mode multichannel
```

The file will be saved in 

```
saves/best_model.pt
```

To test the model, you can use the following command.

```
python main.py --trained_model saves/best_model.pt --mode multichannel
```



## Dataset and Embeddings 

We experiment the model on the following three datasets.

- SST-1: Keep the original splits and train with phrase level dataset and test on sentence level dataset.


## Results

### best dev 
|dataset|rand|static|non-static|multichannel|
|---|---|---|---|---|
|SST-1|43.142598|48.773842|49.137148|49.318801|


### test
|dataset|rand|static|non-static|multichannel|
|---|---|---|---|---|
|SST-1|39.909502|46.380090|45.294118|48.416290|

We do not tune the parameters for each dataset. And the implementation is simplified from the original version on regularization.

