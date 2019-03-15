import os
import models.args


def get_args():
    parser = models.args.get_args()

    parser.add_argument('--resume-snapshot', type=str, default=None)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--bottleneck-layer', action='store_true')
    parser.add_argument('--single-label', action='store_true')
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--mode', type=str, default='static', choices=['rand', 'static', 'non-static'])
    parser.add_argument('--dataset', type=str, default='Reuters', choices=['SST-1', 'SST-2', 'Reuters', 'AAPD', 'IMDB', 'Yelp2014'])
    parser.add_argument('--words-dim', type=int, default=300)
    parser.add_argument('--embed-dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epoch-decay', type=int, default=15)
    parser.add_argument('--word-vectors-dir', default=os.path.join(os.pardir, 'Castor-data', 'embeddings', 'word2vec'))
    parser.add_argument('--word-vectors-file', default='GoogleNews-vectors-negative300.txt')
    parser.add_argument('--trained-model', type=str, default="")
    parser.add_argument('--TAR', type=float, default=0.0, help="temporal activation regularization")
    parser.add_argument('--AR', type=float, default=0.0, help="activation regularization")
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--beta-ema', type=float, default=0, help="temporal averaging")
    parser.add_argument('--wdrop', type=float, default=0.0, help="weight drop")
    parser.add_argument('--embed-droprate', type=float, default=0.0, help="embedding dropout")

    args = parser.parse_args()
    return args