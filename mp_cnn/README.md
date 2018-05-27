# MP-CNN (and "Lite" Variants)

This is a PyTorch reimplementation of the following paper:

+ Hua He, Kevin Gimpel, and Jimmy Lin. [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](http://aclweb.org/anthology/D/D15/D15-1181.pdf). *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP 2015)*, pages 1576-1586.

On top of which we performed additional ablation experiments and explored a number of variants, as described in:

+ Zhucheng Tu. [An Experimental Analysis of Multi-Perspective Convolutional Neural Networks](https://uwspace.uwaterloo.ca/handle/10012/13297). Master's Thesis, Computer Science, University of Waterloo, 2018.

Please ensure you have followed instructions in the main [README](../README.md) doc before running any further commands in this doc.
The commands in this doc assume you are under the root directory of the Castor repo.

## Pre-Trained Models

We provide pre-trained models for SICK, TrecQA, and WikiQA in the [Castor-models](https://git.uwaterloo.ca/jimmylin/Castor-models) repository. They are trained using the commands in each of the dataset sections below; evaluation metrics match the reported values in those sections in our environment.

**SICK** ([`mp_cnn/mpcnn.sick.model`](https://git.uwaterloo.ca/jimmylin/Castor-models/tree/master/mp_cnn))

```
$ python -m mp_cnn ../Castor-models/mp_cnn/mpcnn.sick.model --dataset sick \
    --skip-training
```

**TrecQA** ([`mp_cnn/mpcnn.trecqa.model`](https://git.uwaterloo.ca/jimmylin/Castor-models/tree/master/mp_cnn))

```
$ python -m mp_cnn ../Castor-models/mp_cnn/mpcnn.trecqa.model --dataset trecqa \
    --holistic-filters 200 --skip-training
```

**WikiQA** ([`mp_cnn/mpcnn.wikiqa.model`](https://git.uwaterloo.ca/jimmylin/Castor-models/tree/master/mp_cnn))

```
$ python -m mp_cnn ../Castor-models/mp_cnn/mpcnn.wikiqa.model --dataset wikiqa \
    --holistic-filters 100 --skip-training
```

The commands above assume GPU; for running on the CPU, add `--device -1`.

If you want to train the models yourself, please read on.

## SICK Dataset

To run MP-CNN on the SICK dataset, use the following command. `--dropout 0` is for mimicking the original paper, although adding dropout can improve results. If you have any problems running it check the Troubleshooting section below.

```
python -m mp_cnn mpcnn.sick.model.castor --dataset sick --epochs 19 --dropout 0 --lr 0.0005
```

| Implementation and config        | Pearson's r   | Spearman's p  | MSE        |
| -------------------------------- |:-------------:|:-------------:|:----------:|
| Paper                            | 0.8686        |   0.8047      | 0.2606     |
| PyTorch using above config       | 0.8738        |   0.8116      | 0.2414     |

## TrecQA Dataset

To run MP-CNN on the TrecQA dataset, use the following command:
```
python -m mp_cnn mpcnn.trecqa.model --dataset trecqa --epochs 5 --holistic-filters 200 --lr 0.00018 --regularization 0.0006405 --dropout 0
```

| Implementation and config        | map    | mrr    |
| -------------------------------- |:------:|:------:|
| Paper                            | 0.764  | 0.827  |
| PyTorch using above config       | 0.777  | 0.821  |

This are the TrecQA raw dataset results. The paper results are reported in [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks](https://dl.acm.org/citation.cfm?id=2983872).

## WikiQA Dataset

You also need `trec_eval` for this dataset, similar to TrecQA.

Then, you can run:
```
python -m mp_cnn mpcnn.wikiqa.model --epochs 10 --dataset wikiqa --epochs 5 --holistic-filters 100 --lr 0.00042 --regularization 0.0001683 --dropout 0
```
| Implementation and config        | map    | mrr    |
| -------------------------------- |:------:|:------:|
| Paper                            | 0.693  | 0.709  |
| PyTorch using above config       | 0.717  | 0.729  |

The paper results are reported in [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks](https://dl.acm.org/citation.cfm?id=2983872).

To see all options available, use
```
python -m mp_cnn --help
```

## Optional Dependencies

To optionally visualize the learning curve during training, we make use of https://github.com/lanpa/tensorboard-pytorch to connect to [TensorBoard](https://github.com/tensorflow/tensorboard). These projects require TensorFlow as a dependency, so you need to install TensorFlow before running the commands below. After these are installed, just add `--tensorboard` when running the training commands and open TensorBoard in the browser.

```sh
pip install tensorboardX
pip install tensorflow-tensorboard
```
