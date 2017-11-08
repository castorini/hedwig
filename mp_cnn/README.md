# MP-CNN PyTorch Implementation

This is a PyTorch implementation of the following paper

* Hua He, Kevin Gimpel, and Jimmy Lin. [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](http://aclweb.org/anthology/D/D15/D15-1181.pdf). *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP 2015)*, pages 1576-1586.

The SICK and MSRVID datasets are available in https://github.com/castorini/data, as well as the GloVe word embeddings.

Directory layout should be like this:
```
├── Castor
│   ├── README.md
│   ├── ...
│   └── mp_cnn/
├── data
│   ├── README.md
│   ├── ...
│   ├── msrvid/
│   ├── sick/
│   └── GloVe/
```

## SICK Dataset

To run MP-CNN on the SICK dataset, use the following command. `--dropout 0` is for mimicking the original paper, although adding dropout can improve performance. If you have any problems running it check the Troubleshooting section below.

```
python main.py mpcnn.sick.model.castor --dataset sick --epochs 19 --epsilon 1e-7 --dropout 0
```

| Implementation and config        | Pearson's r   | Spearman's p  |
| -------------------------------- |:-------------:|:-------------:|
| Paper                            | 0.8686        |   0.8047      |
| PyTorch using above config       | 0.8684        |   0.8083      |

## MSRVID Dataset

To run MP-CNN on the MSRVID dataset, use the following command:
```
python main.py mpcnn.msrvid.model.castor --dataset msrvid --batch-size 16 --epsilon 1e-7 --epochs 32 --dropout 0 --regularization 0.0025
```

| Implementation and config        | Pearson's r   |
| -------------------------------- |:-------------:|
| Paper                            | 0.9090        |
| PyTorch using above config       | 0.8911        |

## TrecQA Dataset

To run MP-CNN on (Raw) TrecQA, you first need to run `./get_trec_eval.sh` in `utils` under the repo root while inside the `utils` directory. This will download and compile the official `trec_eval` tool used for evaluation.

Then, you can run:
```
python main.py mpcnn.trecqa.model --dataset trecqa --epochs 5 --regularization 0.0005 --dropout 0.5 --eps 0.1
```

| Implementation and config        | map    | mrr    |
| -------------------------------- |:------:|:------:|
| Paper                            | 0.762  | 0.830  |
| PyTorch using above config       | 0.7904 | 0.8223 |

The paper results are reported in [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks](https://dl.acm.org/citation.cfm?id=2983872).

## WikiQA Dataset

You also need `trec_eval` for this dataset, similar to TrecQA.

Then, you can run:
```
python main.py mpcnn.wikiqa.model --epochs 10 --dataset wikiqa --batch-size 64 --lr 0.0004 --regularization 0.02
```
| Implementation and config        | map    | mrr    |
| -------------------------------- |:------:|:------:|
| Paper                            | 0.693  | 0.709  |
| PyTorch using above config       | 0.693  | 0.7091 |

The paper results are reported in [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks](https://dl.acm.org/citation.cfm?id=2983872).


These are not the optimal hyperparameters but they are decent. This README will be updated with more optimal hyperparameters and results in the future.

To see all options available, use
```
python main.py --help
```

## Troubleshooting

### ModuleNotFoundError: datasets
```
Traceback (most recent call last):
  File "main.py", line 9, in <module>
    from dataset import MPCNNDatasetFactory
  File "/u/z3tu/castorini/Castor/mp_cnn/dataset.py", line 12, in <module>
    from datasets.sick import SICK
ModuleNotFoundError: No module named 'datasets'
```

You need to make sure the repository root is in your `PYTHONPATH` environment variable. One way to do this is while you are in the repo root (Castor) as your current working directory, run `export PYTHONPATH=$(pwd)`.

## Optional Dependencies

To optionally visualize the learning curve during training, we make use of https://github.com/lanpa/tensorboard-pytorch to connect to [TensorBoard](https://github.com/tensorflow/tensorboard). These projects require TensorFlow as a dependency, so you need to install TensorFlow before running the commands below. After these are installed, just add `--tensorboard` when running `main.py` and open TensorBoard in the browser.

```sh
pip install tensorboardX
pip install tensorflow-tensorboard
```
