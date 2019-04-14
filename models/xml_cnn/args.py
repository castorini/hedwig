import os
import models.args


def get_args():
    parser = models.args.get_args()

    parser.add_argument('--mode', type=str, default='multichannel', choices=['rand', 'static', 'non-static', 'multichannel'])
    parser.add_argument('--dataset', type=str, default='Reuters', choices=['Reuters', 'AAPD', 'IMDB', 'Yelp2014'])
    parser.add_argument('--dev-every', type=int, default=30)
    parser.add_argument('--output-channel', type=int, default=100)
    parser.add_argument('--words-dim', type=int, default=300)
    parser.add_argument('--embed-dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epoch-decay', type=int, default=15)
    parser.add_argument('--weight-decay', type=float, default=0)

    parser.add_argument('--num-bottleneck-hidden', type=int, default=512)
    parser.add_argument('--dynamic-pool-length', type=int, default=32)

    parser.add_argument('--word-vectors-dir', default=os.path.join(os.pardir, 'hedwig-data', 'embeddings', 'word2vec'))
    parser.add_argument('--word-vectors-file', default='GoogleNews-vectors-negative300.txt')
    parser.add_argument('--save-path', type=str, default=os.path.join('model_checkpoints', 'xml_cnn'))
    parser.add_argument('--resume-snapshot', type=str)
    parser.add_argument('--trained-model', type=str)

    args = parser.parse_args()
    return args
