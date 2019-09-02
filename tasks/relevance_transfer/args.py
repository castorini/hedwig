import os

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Deep learning models for relevance transfer")
    parser.add_argument('--no-cuda', action='store_false', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--mode', type=str, default='static', choices=['rand', 'static', 'non-static', 'multichannel'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-mult', type=float, default=1)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--dataset', type=str, default='Robust04', choices=['Robust04', 'Robust05', 'Robust45'])
    parser.add_argument('--model', type=str, default='KimCNN', choices=['RegLSTM', 'KimCNN', 'HAN', 'XML-CNN', 'BERT-Base',
                                                                        'BERT-Large', 'HBERT-Base', 'HBERT-Large'])

    parser.add_argument('--dev_every', type=int, default=30)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--save_path', type=str, default=os.path.join('model_checkpoints', 'relevance_transfer'))
    parser.add_argument('--words_dim', type=int, default=300)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epoch_decay', type=int, default=15)
    parser.add_argument('--data_dir', default=os.path.join(os.pardir, 'hedwig-data', 'datasets'))
    parser.add_argument('--word_vectors_dir', default=os.path.join(os.pardir, 'hedwig-data', 'embeddings', 'word2vec'))
    parser.add_argument('--word_vectors_file', help='word vectors filename', default='GoogleNews-vectors-negative300.txt')
    parser.add_argument("--output-path", type=str, default="run.core17.lstm.topics.robust00.txt")
    parser.add_argument('--resume-snapshot', action='store_true')
    parser.add_argument('--resample', action='store_true')

    # RegLSTM parameters
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--tar', action='store_true')
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--beta-ema', type=float, default=0, help="for temporal averaging")
    parser.add_argument('--wdrop', type=float, default=0.0, help="for weight-drop")
    parser.add_argument('--embed-droprate', type=float, default=0.0, help="for embedded dropout")

    # KimCNN parameters
    parser.add_argument('--dropblock', type=float, default=0.0)
    parser.add_argument('--dropblock-size', type=int, default=7)
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--output-channel', type=int, default=100)

    # HAN parameters
    parser.add_argument('--word-num-hidden', type=int, default=50)
    parser.add_argument('--sentence-num-hidden', type=int, default=50)

    # XML-CNN parameters
    parser.add_argument('--bottleneck-layer', action='store_true')
    parser.add_argument('--dynamic-pool', action='store_true')
    parser.add_argument('--variable-dynamic-pool', action='store_true')
    parser.add_argument('--bottleneck-units', type=int, default=100)
    parser.add_argument('--dynamic-pool-length', type=int, default=8)

    # HR-CNN parameters
    parser.add_argument('--sentence-channel', type=int, default=100)

    # BERT parameters
    parser.add_argument('--cache-dir', default='cache', type=str)
    parser.add_argument('--variant', type=str, choices=['bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased'])
    parser.add_argument('--max-seq-length', default=128, type=int)
    parser.add_argument('--max-doc-length', default=16, type=int)
    parser.add_argument('--warmup-proportion', default=0.1, type=float)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--loss-scale', type=float, default=0)

    # Re-ranking parameters
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument("--ret-ranks", type=str, help='retrieval rank file', default="run.core17.bm25+rm3.wcro0405.hits10000.txt")
    parser.add_argument("--clf-ranks", type=str, help='classification rank file', default="run.core17.lstm.topics.robust45.txt")

    args = parser.parse_args()
    return args
