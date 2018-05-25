import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MPCNN(nn.Module):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        super(MPCNN, self).__init__()
        self.arch = 'mpcnn'
        self.n_word_dim = n_word_dim
        self.n_holistic_filters = n_holistic_filters
        self.n_per_dim_filters = n_per_dim_filters
        self.filter_widths = filter_widths
        self.ext_feats = ext_feats
        self.attention = attention
        self.wide_conv = wide_conv

        self.in_channels = n_word_dim if attention == 'none' else 2 * n_word_dim

        self._add_layers()

        # compute number of inputs to first hidden layer
        n_feats = self._get_n_feats()

        self.final_layers = nn.Sequential(
            nn.Linear(n_feats, hidden_layer_units),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_units, num_classes),
            nn.LogSoftmax(1)
        )

    def _add_layers(self):
        holistic_conv_layers_max = []
        holistic_conv_layers_min = []
        holistic_conv_layers_mean = []
        per_dim_conv_layers_max = []
        per_dim_conv_layers_min = []

        for ws in self.filter_widths:
            if np.isinf(ws):
                continue

            padding = ws-1 if self.wide_conv else 0

            holistic_conv_layers_max.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.n_holistic_filters, ws, padding=padding),
                nn.Tanh()
            ))

            holistic_conv_layers_min.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.n_holistic_filters, ws, padding=padding),
                nn.Tanh()
            ))

            holistic_conv_layers_mean.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.n_holistic_filters, ws, padding=padding),
                nn.Tanh()
            ))

            per_dim_conv_layers_max.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.in_channels * self.n_per_dim_filters, ws, padding=padding, groups=self.in_channels),
                nn.Tanh()
            ))

            per_dim_conv_layers_min.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.in_channels * self.n_per_dim_filters, ws, padding=padding, groups=self.in_channels),
                nn.Tanh()
            ))

        self.holistic_conv_layers_max = nn.ModuleList(holistic_conv_layers_max)
        self.holistic_conv_layers_min = nn.ModuleList(holistic_conv_layers_min)
        self.holistic_conv_layers_mean = nn.ModuleList(holistic_conv_layers_mean)
        self.per_dim_conv_layers_max = nn.ModuleList(per_dim_conv_layers_max)
        self.per_dim_conv_layers_min = nn.ModuleList(per_dim_conv_layers_min)

    def _get_n_feats(self):
        COMP_1_COMPONENTS_HOLISTIC, COMP_1_COMPONENTS_PER_DIM, COMP_2_COMPONENTS = 2 + self.n_holistic_filters, 2 + self.in_channels, 2
        n_feats_h = 3 * self.n_holistic_filters * COMP_2_COMPONENTS
        n_feats_v = (
            # comparison units from holistic conv for min, max, mean pooling for non-infinite widths
            3 * ((len(self.filter_widths) - 1) ** 2) * COMP_1_COMPONENTS_HOLISTIC +
            # comparison units from holistic conv for min, max, mean pooling for infinite widths
            3 * 3 +
            # comparison units from per-dim conv
            2 * (len(self.filter_widths) - 1) * self.n_per_dim_filters * COMP_1_COMPONENTS_PER_DIM
        )
        n_feats = n_feats_h + n_feats_v + self.ext_feats
        return n_feats

    def _get_blocks_for_sentence(self, sent):
        block_a = {}
        block_b = {}
        for ws in self.filter_widths:
            if np.isinf(ws):
                sent_flattened, sent_flattened_size = sent.contiguous().view(sent.size(0), 1, -1), sent.size(1) * sent.size(2)
                block_a[ws] = {
                    'max': F.max_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1),
                    'min': F.max_pool1d(-1 * sent_flattened, sent_flattened_size).view(sent.size(0), -1),
                    'mean': F.avg_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1)
                }
                continue

            holistic_conv_out_max = self.holistic_conv_layers_max[ws - 1](sent)
            holistic_conv_out_min = self.holistic_conv_layers_min[ws - 1](sent)
            holistic_conv_out_mean = self.holistic_conv_layers_mean[ws - 1](sent)
            block_a[ws] = {
                'max': F.max_pool1d(holistic_conv_out_max, holistic_conv_out_max.size(2)).contiguous().view(-1, self.n_holistic_filters),
                'min': F.max_pool1d(-1 * holistic_conv_out_min, holistic_conv_out_min.size(2)).contiguous().view(-1, self.n_holistic_filters),
                'mean': F.avg_pool1d(holistic_conv_out_mean, holistic_conv_out_mean.size(2)).contiguous().view(-1, self.n_holistic_filters)
            }

            per_dim_conv_out_max = self.per_dim_conv_layers_max[ws - 1](sent)
            per_dim_conv_out_min = self.per_dim_conv_layers_min[ws - 1](sent)
            block_b[ws] = {
                'max': F.max_pool1d(per_dim_conv_out_max, per_dim_conv_out_max.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters),
                'min': F.max_pool1d(-1 * per_dim_conv_out_min, per_dim_conv_out_min.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters)
            }
        return block_a, block_b

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        for pool in ('max', 'min', 'mean'):
            regM1, regM2 = [], []
            for ws in self.filter_widths:
                x1 = sent1_block_a[ws][pool].unsqueeze(2)
                x2 = sent2_block_a[ws][pool].unsqueeze(2)
                if np.isinf(ws):
                    x1 = x1.expand(-1, self.n_holistic_filters, -1)
                    x2 = x2.expand(-1, self.n_holistic_filters, -1)
                regM1.append(x1)
                regM2.append(x2)

            regM1 = torch.cat(regM1, dim=2)
            regM2 = torch.cat(regM2, dim=2)

            # Cosine similarity
            comparison_feats.append(F.cosine_similarity(regM1, regM2, dim=2))
            # Euclidean distance
            pairwise_distances = []
            for x1, x2 in zip(regM1, regM2):
                dist = F.pairwise_distance(x1, x2).view(1, -1)
                pairwise_distances.append(dist)
            comparison_feats.append(torch.cat(pairwise_distances))

        return torch.cat(comparison_feats, dim=1)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b):
        comparison_feats = []
        ws_no_inf = [w for w in self.filter_widths if not np.isinf(w)]
        for pool in ('max', 'min', 'mean'):
            for ws1 in self.filter_widths:
                x1 = sent1_block_a[ws1][pool]
                for ws2 in self.filter_widths:
                    x2 = sent2_block_a[ws2][pool]
                    if (not np.isinf(ws1) and not np.isinf(ws2)) or (np.isinf(ws1) and np.isinf(ws2)):
                        comparison_feats.append(F.cosine_similarity(x1, x2).unsqueeze(1))
                        comparison_feats.append(F.pairwise_distance(x1, x2).unsqueeze(1))
                        comparison_feats.append(torch.abs(x1 - x2))

        for pool in ('max', 'min'):
            for ws in ws_no_inf:
                oG_1B = sent1_block_b[ws][pool]
                oG_2B = sent2_block_b[ws][pool]
                for i in range(0, self.n_per_dim_filters):
                    x1 = oG_1B[:, :, i]
                    x2 = oG_2B[:, :, i]
                    comparison_feats.append(F.cosine_similarity(x1, x2).unsqueeze(1))
                    comparison_feats.append(F.pairwise_distance(x1, x2).unsqueeze(1))
                    comparison_feats.append(torch.abs(x1 - x2))

        return torch.cat(comparison_feats, dim=1)

    def concat_attention(self, sent1, sent2, word_to_doc_count=None, raw_sent1=None, raw_sent2=None):
        sent1_transposed = sent1.transpose(1, 2)
        attention_dot = torch.bmm(sent1_transposed, sent2)
        sent1_norms = torch.norm(sent1_transposed, p=2, dim=2, keepdim=True)
        sent2_norms = torch.norm(sent2, p=2, dim=1, keepdim=True)
        attention_norms = torch.bmm(sent1_norms, sent2_norms)
        attention_matrix = attention_dot / attention_norms

        if self.attention == 'idf' and word_to_doc_count is not None:
            idf_matrix1 = sent1.data.new_ones(sent1.size(0), sent1.size(2))
            for i, sent in enumerate(raw_sent1):
                for j, word in enumerate(sent.split(' ')):
                    idf_matrix1[i, j] /= word_to_doc_count.get(word, 1)

            idf_matrix2 = sent2.data.new_ones(sent2.size(0), sent2.size(2)).fill_(1)
            for i, sent in enumerate(raw_sent2):
                for j, word in enumerate(sent.split(' ')):
                    idf_matrix2[i, j] /= word_to_doc_count.get(word, 1)

            sum_row = (attention_matrix * idf_matrix2.unsqueeze(1)).sum(2)
            sum_col = (attention_matrix * idf_matrix1.unsqueeze(2)).sum(1)
        else:
            sum_row = attention_matrix.sum(2)
            sum_col = attention_matrix.sum(1)

        if self.attention == 'idf' and word_to_doc_count is not None:
            for i, sent in enumerate(raw_sent1):
                for j, word in enumerate(sent.split(' ')):
                    sum_row[i, j] /= word_to_doc_count.get(word, 1)

            for i, sent in enumerate(raw_sent2):
                for j, word in enumerate(sent.split(' ')):
                    sum_col[i, j] /= word_to_doc_count.get(word, 1)

        attention_weight_vec1 = F.softmax(sum_row, 1)
        attention_weight_vec2 = F.softmax(sum_col, 1)

        attention_weighted_sent1 = attention_weight_vec1.unsqueeze(1).expand(-1, self.n_word_dim, -1) * sent1
        attention_weighted_sent2 = attention_weight_vec2.unsqueeze(1).expand(-1, self.n_word_dim, -1) * sent2
        attention_emb1 = torch.cat((attention_weighted_sent1, sent1), dim=1)
        attention_emb2 = torch.cat((attention_weighted_sent2, sent2), dim=1)
        return attention_emb1, attention_emb2

    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None):
        # Attention
        if self.attention != 'none':
            sent1, sent2 = self.concat_attention(sent1, sent2, word_to_doc_count, raw_sent1, raw_sent2)

        # Sentence modelling module
        sent1_block_a, sent1_block_b = self._get_blocks_for_sentence(sent1)
        sent2_block_a, sent2_block_b = self._get_blocks_for_sentence(sent2)

        # Similarity measurement layer
        feat_h = self._algo_1_horiz_comp(sent1_block_a, sent2_block_a)
        feat_v = self._algo_2_vert_comp(sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b)
        combined_feats = [feat_h, feat_v, ext_feats] if self.ext_feats else [feat_h, feat_v]
        feat_all = torch.cat(combined_feats, dim=1)

        preds = self.final_layers(feat_all)
        return preds
