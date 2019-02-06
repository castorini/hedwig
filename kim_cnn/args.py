import os

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Kim CNN")
    parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0, help='Use -1 for CPU')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--mode', type=str, default='multichannel', choices=['rand', 'static', 'non-static', 'multichannel'])
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--dataset', type=str, default='SST-1', choices=['SST-1', 'SST-2', 'Reuters', 'AAPD', 'IMDB', 'Yelp2014'])
    parser.add_argument('--resume_snapshot', type=str, default=None)
    parser.add_argument('--single_label', action='store_true')
    parser.add_argument('--dev_every', type=int, default=30)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--save_path', type=str, default='kim_cnn/saves')
    parser.add_argument('--output_channel', type=int, default=100)
    parser.add_argument('--words_dim', type=int, default=300)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epoch_decay', type=int, default=15)
    parser.add_argument('--data_dir', help='word vectors directory',
                        default=os.path.join(os.pardir, 'Castor-data', 'datasets'))
    parser.add_argument('--word_vectors_dir', help='word vectors directory',
                        default=os.path.join(os.pardir, 'Castor-data', 'embeddings', 'word2vec'))
    parser.add_argument('--word_vectors_file', help='word vectors filename', default='GoogleNews-vectors-negative300.txt')
    parser.add_argument('--trained_model', type=str, default="")
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--onnx', action='store_true', default=False, help='Export model in ONNX format')
    parser.add_argument('--onnx_batch_size', type=int, default=1024, help='Batch size for ONNX export')
    parser.add_argument('--onnx_sent_len', type=int, default=32, help='Sentence length for ONNX export')

    args = parser.parse_args()
    return args
