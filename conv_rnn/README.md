## Convolutional RNN

Implementation based on [[1]](http://dl.acm.org/citation.cfm?id=3098140).

### Usage

Run `./getData.sh` to fetch the data. The project structure should now look like this:

```
├── conv_rnn/
│   ├── data/
│   ├── saves/
│   └── *.*
```
You may then run `python train.py` and `python test.py` for training and testing, respectively. For more options, add the `-h` switch.

### Empirical results
Best dev | Test
-- | --
52.04359673024523 | 50.85972850678733

### References
[1] Chenglong Wang, Feijun Jiang, and Hongxia Yang. 2017. A Hybrid Framework for Text Modeling with Convolutional RNN. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '17).
