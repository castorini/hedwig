# DecAtt

This is a PyTorch reimplementation of the following paper:

```
@inproceedings{parikh-EtAl:2016:EMNLP2016,
  author     = {Parikh, Ankur  and  T\"{a}ckstr\"{o}m, Oscar  and  Das, Dipanjan  and  Uszkoreit, Jakob},
  title    = {A Decomposable Attention Model for Natural Language Inference},
booktitle  = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year     = {2016}
} 
```


Please ensure you have followed instructions in the main [README](../README.md) doc before running any further commands in this doc.
The commands in this doc assume you are under the root directory of the Castor repo.

## SICK Dataset

To run DecAtt on the SICK dataset, use the following command. `--dropout 0` is for mimicking the original paper, although adding dropout can improve results. If you have any problems running it check the Troubleshooting section below.

```
python -m decatt decatt.sick.model --dataset sick --epochs 500 --regularization 5e-4 --lr 0.001 --lr-reduce-factor 0.5 --dropout 0.1
```

| Implementation and config        | Pearson's r   | Spearman's p  | MSE        |
| -------------------------------- |:-------------:|:-------------:|:----------:|
| PyTorch using above config       |  0.80094564      |   0.7184082390455326     |  0.3711671233177185  |

## TrecQA Dataset

To run DecAtt on the TrecQA dataset, use the following command:
```
python -m decatt decatt.trecqa.model --dataset trecqa --epochs 500 --regularization 5e-4 --lr 0.001 --lr-reduce-factor 0.5 --dropout 0.1
```

| Implementation and config        | map    | mrr    |
| -------------------------------- |:------:|:------:|
| PyTorch using above config       | 0.6536  | 0.6848 |

This are the TrecQA raw dataset results. The paper results are reported in [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks](https://dl.acm.org/citation.cfm?id=2983872).

## WikiQA Dataset

You also need `trec_eval` for this dataset, similar to TrecQA.

Then, you can run:
```
python -m decatt decatt.wikiqa.model --dataset wikiqa --epochs 500 --regularization 5e-4 --lr 0.001 --lr-reduce-factor 0.5 --dropout 0.1
```
| Implementation and config        | map    | mrr    |
| -------------------------------- |:------:|:------:|
| PyTorch using above config       | 0.6462  | 0.6603  |


To see all options available, use
```
python -m decatt --help
```

## Optional Dependencies

To optionally visualize the learning curve during training, we make use of https://github.com/lanpa/tensorboard-pytorch to connect to [TensorBoard](https://github.com/tensorflow/tensorboard). These projects require TensorFlow as a dependency, so you need to install TensorFlow before running the commands below. After these are installed, just add `--tensorboard` when running the training commands and open TensorBoard in the browser.

```sh
pip install tensorboardX
pip install tensorflow-tensorboard
```
