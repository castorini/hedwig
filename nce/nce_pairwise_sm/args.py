from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="SM CNN")
    parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
    parser.add_argument('--gpu', type=int, default=-1) # Use -1 for CPU
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mode', type=str, default='static')
    parser.add_argument('--lr', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--dataset', type=str, default='trec')
    parser.add_argument('--resume_snapshot', type=str, default=None)
    parser.add_argument('--dev_every', type=int, default=100)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--save_path', type=str,    default='saves')
    parser.add_argument('--output_channel', type=int, default=150)
    parser.add_argument('--filter_width', type=int, default=5)
    parser.add_argument('--words_dim', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epoch_decay', type=int, default=15)
    parser.add_argument('--wordvec_dir', type=str, default='../../../Castor-data/embeddings/word2vec/')
    parser.add_argument('--vector_cache', type=str, default='word2vec.trecqa.pt')
    parser.add_argument('--trained_model', type=str, default="")
    parser.add_argument('--weight_decay',type=float, default=1e-5)
    parser.add_argument('--ext_feats_size', type=int, default=4)
    parser.add_argument('--neg_num', type=int, default=5)
    parser.add_argument('--neg_sample', type=str, default="random")
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--optimizer', type=str, default="adadelta")
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='use TensorBoard to visualize training (default: false)')

    args = parser.parse_args()
    return args
