# SSE

This is a PyTorch reimplementation of the following paper:

```
@InProceedings{nie-bansal:2017:RepEval,
 author    = {Nie, Yixin  and  Bansal, Mohit},
 title     = {Shortcut-Stacked Sentence Encoders for Multi-Domain Inference},
 booktitle = {Proceedings of the 2nd Workshop on Evaluating Vector Space Representations for NLP},
 year      = {2017}
}
```


Please ensure you have followed instructions in the main [README](../README.md) doc before running any further commands in this doc.
The commands in this doc assume you are under the root directory of the Castor repo.

## SICK Dataset

To run SSE on the SICK dataset, use the following command. `--dropout 0` is for mimicking the original paper, although adding dropout can improve results. If you have any problems running it check the Troubleshooting section below.

```
python -m sse sse.sick.model.castor --dataset sick --epochs 19 --dropout 0.5 --lr 0.0002 --regularization 1e-4 
```

| Implementation and config        | Pearson's r   | Spearman's p  | MSE        |
| -------------------------------- |:-------------:|:-------------:|:----------:|
| PyTorch using above config       |  0.8812158      | 0.8292130938075161       |  0.22950001060962677    |

## TrecQA Dataset

To run SSE on the TrecQA dataset, use the following command:
```
python -m sse sse.trecqa.model --dataset trecqa --epochs 5 --holistic-filters 200 --lr 0.00018 --regularization 0.0006405 --dropout 0
```

| Implementation and config        | map    | mrr    |
| -------------------------------- |:------:|:------:|
| PyTorch using above config       |   |   |

This are the TrecQA raw dataset results. The paper results are reported in [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks](https://dl.acm.org/citation.cfm?id=2983872).

## WikiQA Dataset

You also need `trec_eval` for this dataset, similar to TrecQA.

Then, you can run:
```
python -m sse sse.wikiqa.model --epochs 10 --dataset wikiqa --epochs 5 --holistic-filters 100 --lr 0.00042 --regularization 0.0001683 --dropout 0
```
| Implementation and config        | map    | mrr    |
| -------------------------------- |:------:|:------:|
| PyTorch using above config       |   |   |


To see all options available, use
```
python -m sse --help
```

## Optional Dependencies

To optionally visualize the learning curve during training, we make use of https://github.com/lanpa/tensorboard-pytorch to connect to [TensorBoard](https://github.com/tensorflow/tensorboard). These projects require TensorFlow as a dependency, so you need to install TensorFlow before running the commands below. After these are installed, just add `--tensorboard` when running the training commands and open TensorBoard in the browser.

```sh
pip install tensorboardX
pip install tensorflow-tensorboard
```
