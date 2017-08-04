import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MPCNN(nn.Module):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes):
        super(MPCNN, self).__init__()

        self.n_word_dim = n_word_dim
        self.n_per_dim_filters = n_per_dim_filters
        self.filter_widths = filter_widths
        holistic_conv_layers = []
        per_dim_conv_layers = []

        for ws in filter_widths:
            if np.isinf(ws):
                continue

            holistic_conv_layers.append(nn.Sequential(
                nn.Conv1d(n_word_dim, n_holistic_filters, ws),
                nn.Tanh()
            ))

            per_dim_conv_layers.append(nn.Sequential(
                nn.Conv1d(n_word_dim, n_word_dim * n_per_dim_filters, ws, groups=n_word_dim),
                nn.Tanh()
            ))

        self.holistic_conv_layers = nn.ModuleList(holistic_conv_layers)
        self.per_dim_conv_layers = nn.ModuleList(per_dim_conv_layers)

        # compute number of inputs to first hidden layer
        COMP_1_COMPONENTS, COMP_2_COMPONENTS = 2 + n_word_dim, 2
        n_feat_h = 3 * len(self.filter_widths) * COMP_2_COMPONENTS
        n_feat_v = 3 * (len(self.filter_widths) ** 2) * COMP_1_COMPONENTS + 2 * (len(self.filter_widths) - 1) * n_per_dim_filters * COMP_1_COMPONENTS
        n_feat = n_feat_h + n_feat_v

        self.final_layers = nn.Sequential(
            nn.Linear(n_feat, hidden_layer_units),
            nn.Tanh(),
            nn.Linear(hidden_layer_units, num_classes),
            nn.LogSoftmax()
        )

    def _get_blocks_for_sentence(self, sent):
        block_a = {}
        block_b = {}
        for ws in self.filter_widths:
            holistic_conv_out = self.holistic_conv_layers[ws - 1](sent) if not np.isinf(ws) else sent
            block_a[ws] = {
                'max': F.max_pool1d(holistic_conv_out, holistic_conv_out.size()[2]).view(-1, self.n_word_dim),
                'min': F.max_pool1d(-1 * holistic_conv_out, holistic_conv_out.size()[2]).view(-1, self.n_word_dim),
                'mean': F.avg_pool1d(holistic_conv_out, holistic_conv_out.size()[2]).view(-1, self.n_word_dim)
            }

            # only compute per-dimension convolution for non-infinity widths
            if np.isinf(ws):
                continue

            per_dim_conv_out = self.per_dim_conv_layers[ws - 1](sent)
            block_b[ws] = {
                'max': F.max_pool1d(per_dim_conv_out, per_dim_conv_out.size()[2]).view(-1, self.n_word_dim, self.n_per_dim_filters),
                'min': F.max_pool1d(-1 * per_dim_conv_out, per_dim_conv_out.size()[2]).view(-1, self.n_word_dim, self.n_per_dim_filters)
            }
        return block_a, block_b

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        for pool in ('max', 'min', 'mean'):
            for ws in self.filter_widths:
                x1 = sent1_block_a[ws][pool]
                x2 = sent2_block_a[ws][pool]
                batch_size = x1.size()[0]
                comparison_feats.append(F.cosine_similarity(x1, x2).view(batch_size, 1))
                comparison_feats.append(F.pairwise_distance(x1, x2))
        return torch.cat(comparison_feats, dim=1)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b):
        comparison_feats = []
        for pool in ('max', 'min', 'mean'):
            for ws1 in self.filter_widths:
                x1 = sent1_block_a[ws1][pool]
                batch_size = x1.size()[0]
                for ws2 in self.filter_widths:
                    x2 = sent2_block_a[ws2][pool]
                    comparison_feats.append(F.cosine_similarity(x1, x2).view(batch_size, 1))
                    comparison_feats.append(F.pairwise_distance(x1, x2))
                    comparison_feats.append(torch.abs(x1 - x2))

        for pool in ('max', 'min'):
            ws_no_inf = [w for w in self.filter_widths if not np.isinf(w)]
            for ws in ws_no_inf:
                oG_1B = sent1_block_b[ws][pool]
                oG_2B = sent2_block_b[ws][pool]
                for i in range(0, self.n_per_dim_filters):
                    x1 = oG_1B[:, :, i]
                    x2 = oG_2B[:, :, i]
                    comparison_feats.append(F.cosine_similarity(x1, x2).view(batch_size, 1))
                    comparison_feats.append(F.pairwise_distance(x1, x2))
                    comparison_feats.append(torch.abs(x1 - x2))

        return torch.cat(comparison_feats, dim=1)

    def forward(self, sent1, sent2):
        # Sentence modeling module
        sent1_block_a, sent1_block_b = self._get_blocks_for_sentence(sent1)
        sent2_block_a, sent2_block_b = self._get_blocks_for_sentence(sent2)

        # Similarity measurement layer
        feat_h = self._algo_1_horiz_comp(sent1_block_a, sent2_block_a)
        feat_v = self._algo_2_vert_comp(sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b)
        feat_all = torch.cat([feat_h, feat_v], dim=1)

        preds = self.final_layers(feat_all)
        return preds
