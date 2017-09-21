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

To run MP-CNN on the SICK dataset, use the following command. `--dropout 0` is for mimicking the original paper, although adding dropout can improve performance.

```
python main.py mpcnn.sick.model.castor --dataset sick --epochs 19 --epsilon 1e-7 --dropout 0
```

| Implementation and config        | Pearson's r   | Spearman's p  |
| -------------------------------- |:-------------:|:-------------:|
| Paper                            | 0.8686        |   0.8047      |
| PyTorch using above config       | 0.8763        |   0.8215      |

To run MP-CNN on the MSRVID dataset, use the following command:
```
python main.py mpcnn.msrvid.model.castor --dataset msrvid --batch-size 16 --epsilon 1e-7 --epochs 32 --dropout 0 --regularization 0.0025
```

| Implementation and config        | Pearson's r   |
| -------------------------------- |:-------------:|
| Paper                            | 0.9090        |
| PyTorch using above config       | 0.9050        |

These are not the optimal hyperparameters but they are decent. This README will be updated with more optimal hyperparameters and results in the future.

To see all options available, use
```
python main.py --help
```
