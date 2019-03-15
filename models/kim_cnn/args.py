import os

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Kim CNN")
    parser.add_argument('--no-cuda', action='store_false', help='do not use cuda', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0, help='Use -1 for CPU')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--mode', type=str, default='multichannel', choices=['rand', 'static', 'non-static', 'multichannel'])
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--dataset', type=str, default='SST-1', choices=['SST-1', 'SST-2', 'Reuters', 'AAPD', 'IMDB', 'Yelp2014'])
    parser.add_argument('--resume-snapshot', type=str, default=None)
    parser.add_argument('--single-label', action='store_true')
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--save-path', type=str, default='kim_cnn/saves')
    parser.add_argument('--output-channel', type=int, default=100)
    parser.add_argument('--words-dim', type=int, default=300)
    parser.add_argument('--embed-dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epoch-decay', type=int, default=15)
    parser.add_argument('--data-dir', help='word vectors directory',
                        default=os.path.join(os.pardir, 'Castor-data', 'datasets'))
    parser.add_argument('--word-vectors-dir', help='word vectors directory',
                        default=os.path.join(os.pardir, 'Castor-data', 'embeddings', 'word2vec'))
    parser.add_argument('--word-vectors-file', help='word vectors filename', default='GoogleNews-vectors-negative300.txt')
    parser.add_argument('--trained-model', type=str, default="")
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--onnx', action='store_true', default=False, help='Export model in ONNX format')
    parser.add_argument('--onnx-batch-size', type=int, default=1024, help='Batch size for ONNX export')
    parser.add_argument('--onnx-sent-len', type=int, default=32, help='Sentence length for ONNX export')

    args = parser.parse_args()
    return args
