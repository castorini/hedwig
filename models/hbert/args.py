import os

import models.args


def get_args():
    parser = models.args.get_args()

    parser.add_argument('--model', default=None, type=str, required=True)
    parser.add_argument('--dataset', type=str, default='SST-2', choices=['SST-2', 'AGNews', 'Reuters', 'AAPD', 'IMDB', 'Yelp2014'])
    parser.add_argument('--save-path', type=str, default=os.path.join('model_checkpoints', 'bert'))
    parser.add_argument('--cache-dir', default='cache', type=str)
    parser.add_argument('--trained-model', default=None, type=str)
    parser.add_argument('--local-rank', type=int, default=-1, help='local rank for distributed training')
    parser.add_argument('--fp16', action='store_true', help='enable 16-bit floating point precision')
    parser.add_argument('--loss-scale', type=float, default=0, help='loss scaling to improve fp16 numeric stability')

    parser.add_argument('--lr-mult', type=float, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--dropblock', type=float, default=0.0)
    parser.add_argument('--dropblock-size', type=int, default=7)
    parser.add_argument('--beta-ema', type=float, default=0)
    parser.add_argument('--embed-droprate', type=float, default=0.0)
    parser.add_argument('--dynamic-pool', action='store_true')
    parser.add_argument('--dynamic-pool-length', type=int, default=8)
    parser.add_argument('--output-channel', type=int, default=100)

    parser.add_argument('--max-seq-length', default=128, type=int,
                        help='maximum total input sequence length after tokenization')

    parser.add_argument('--max-doc-length', default=16, type=int,
                        help='maximum number of lines processed in one document')

    parser.add_argument('--warmup-proportion', default=0.1, type=float,
                        help='proportion of training to perform linear learning rate warmup for')

    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='number of updates steps to accumulate before performing a backward/update pass')

    args = parser.parse_args()
    return args
