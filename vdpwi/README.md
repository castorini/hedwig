# VDPWI PyTorch Implementation

This is a PyTorch implementation of the following paper

* Hua He and Jimmy Lin. [Pairwise Word Interaction Modeling with Deep Neural Networks for Semantic Similarity Measurement.](http://www.aclweb.org/anthology/N16-1108) *Proceedings of the 15th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL/HLT 2016)*, pages 937-948.


Please ensure you have followed instructions in the main [README](../README.md) doc before running any further commands in this doc.

## SICK Dataset

To run VDPWI on the SICK dataset, use the following command. If you have any problems running it check the Troubleshooting section below.

```
python -m vdpwi vdpwi.sick.model.castor --dataset sick --epochs 19 --epsilon 1e-7
```

## MSRVID Dataset

To run VDPWI on the MSRVID dataset, use the following command:
```
python -m vdpwi vdpwi.msrvid.model.castor --dataset msrvid --batch-size 16 --epochs 32 --regularization 0.0025
```

## TrecQA Dataset

To run VDPWI on (Raw) TrecQA, you first need to run `./get_trec_eval.sh` in `utils` under the repo root while inside the `utils` directory. This will download and compile the official `trec_eval` tool used for evaluation.

Then, you can run:
```
python -m vdpwi vdpwi.trecqa.model --dataset trecqa --epochs 5 --regularization 0.0005 --eps 0.1
```

The paper results are reported in [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks](https://dl.acm.org/citation.cfm?id=2983872).

## WikiQA Dataset

You also need `trec_eval` for this dataset, similar to TrecQA.

Then, you can run:
```
python -m vdpwi vdpwi.wikiqa.model --epochs 10 --dataset wikiqa --batch-size 64 --lr 0.0004 --regularization 0.02
```

To see all options available, use
```
python -m vdpwi --help
```
