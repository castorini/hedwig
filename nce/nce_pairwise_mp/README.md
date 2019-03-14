# MP-CNN PyTorch Implementation

This is a PyTorch implementation of the following paper

* Hua He, Kevin Gimpel, and Jimmy Lin. [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](http://aclweb.org/anthology/D/D15/D15-1181.pdf). *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP 2015)*, pages 1576-1586.
* Jinfeng Rao, Hua He, and Jimmy Lin. [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks.](http://dl.acm.org/citation.cfm?id=2983872) *Proceedings of the 25th ACM International on Conference on Information and Knowledge Management (CIKM 2016)*, pages 1913-1916.

Please ensure you have followed instructions in the main [README](../README.md) doc before running any further commands in this doc.

## TrecQA Dataset

To run MP-CNN on (Raw) TrecQA, you first need to run `./get_trec_eval.sh` in `utils` under the repo root while inside the `utils` directory. This will download and compile the official `trec_eval` tool used for evaluation.

Then, you can run:
```
python train_script.py --dataset wikiqa --device -1
```

Metric|Without NCE (original paper) | only random sampling | only max sampling | Pair-wise+nagative sampling (original paper) | Pair-wise+random sampling| Pair-wise+nagative sampling | Pair-wise+nagative sampling+pair weighting
-------|------|----------|------------|------------|------------|------|------
MAP    |0.762 | 0.7579| 0.7678|0.780  | 0.7745   |0.7873|0.7683
MRR    |0.830 |0.8239| 0.8387|0.834  | 0.8435  |0.8414|0.8253

The paper results are reported in [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks](https://dl.acm.org/citation.cfm?id=2983872).

## WikiQA Dataset

You also need `trec_eval` for this dataset, similar to TrecQA.

Then, you can run:
```
python train_script.py --dataset trecqa --device -1
```

Metric|Without NCE (original paper) | only random sampling  | only max sampling| Pair-wise+nagative sampling (original paper)| Pair-wise+random sampling | Pair-wise+nagative sampling | Pair-wise+nagative sampling+pair weighting
-------|-------|------|----------|------------|------------|------------|------------
MAP    |0.693 | 0.6744| 0.6795 | 0.701| 0.7047    |0.7049| 0.7047
MRR    |0.709 | 0.6898| 0.6951 |0.718 | 0.7172   |0.7192| 0.7211


The paper results are reported in [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks](https://dl.acm.org/citation.cfm?id=2983872).


To see all options available and train with your parameters, use
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
