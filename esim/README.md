# ESIM

This is a PyTorch reimplementation of the following paper:

```
@InProceedings{Chen-Qian:2017:ACL,
 author    = {Chen, Qian and Zhu, Xiaodan and Ling, Zhenhua and Wei, Si and Jiang, Hui and Inkpen, Diana},
 title     = {Enhanced {LSTM} for Natural Language Inference},
 booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL)},
 year      = {2017}
}
```


Please ensure you have followed instructions in the main [README](../README.md) doc before running any further commands in this doc.
The commands in this doc assume you are under the root directory of the Castor repo.

## SICK Dataset

To run ESIM on the SICK dataset, use the following command. `--dropout 0` is for mimicking the original paper, although adding dropout can improve results. If you have any problems running it check the Troubleshooting section below.

```
python -m esim esim.sick.model_tune --dataset sick --epochs 25 --regularization 1e-4 --lr 0.001 --batch-size 64 --lr-reduce-factor 0.3 --dropout 0.2
```

| Implementation and config        | Pearson's r   | Spearman's p  | MSE        |
| -------------------------------- |:-------------:|:-------------:|:----------:|
| PyTorch using above config       |     0.878273    |   0.823042214423      |  0.25375571846961975    |

## TrecQA Dataset

To run ESIM on the TrecQA dataset, use the following command:
```
python -m esim esim.trecqa.model --dataset trecqa --epochs 5 --holistic-filters 200 --lr 0.00018 --regularization 0.0006405 --dropout 0
```

| Implementation and config        | map    | mrr    |
| -------------------------------- |:------:|:------:|
| PyTorch using above config       |   |   |

This are the TrecQA raw dataset results. The paper results are reported in [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks](https://dl.acm.org/citation.cfm?id=2983872).

## WikiQA Dataset

You also need `trec_eval` for this dataset, similar to TrecQA.

Then, you can run:
```
python -m esim esim.wikiqa.model --epochs 10 --dataset wikiqa --epochs 5 --holistic-filters 100 --lr 0.00042 --regularization 0.0001683 --dropout 0
```
| Implementation and config        | map    | mrr    |
| -------------------------------- |:------:|:------:|
| PyTorch using above config       |   |   |


To see all options available, use
```
python -m esim --help
```

## Optional Dependencies

To optionally visualize the learning curve during training, we make use of https://github.com/lanpa/tensorboard-pytorch to connect to [TensorBoard](https://github.com/tensorflow/tensorboard). These projects require TensorFlow as a dependency, so you need to install TensorFlow before running the commands below. After these are installed, just add `--tensorboard` when running the training commands and open TensorBoard in the browser.

```sh
pip install tensorboardX
pip install tensorflow-tensorboard
```
