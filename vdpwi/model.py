import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hard_pad2d(x, pad):
    def pad_side(idx):
        pad_len = max(pad - x.size(idx), 0)
        return [0, pad_len]
    padding = pad_side(3)
    padding.extend(pad_side(2))
    x = F.pad(x, padding)
    return x[:, :, :pad, :pad]

class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layers = config['res_layers']
        n_maps = config['res_fmaps']
        n_labels = config['n_labels']
        self.conv0 = nn.Conv2d(12, n_maps, (3, 3), padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(n_maps, n_maps, (3, 3), padding=1) for _ in range(n_layers)])
        self.output = nn.Linear(n_maps, n_labels)
        self.input_len = None

    def forward(self, x):
        x = F.relu(self.conv0(x))
        old_x = x
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x))
            if i % 2 == 1:
                x += old_x
                old_x = x
        x = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        return F.log_softmax(self.output(x), 1)

class VDPWIConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        def make_conv(n_in, n_out):
            conv = nn.Conv2d(n_in, n_out, 3, padding=1)
            conv.bias.data.zero_()
            nn.init.xavier_normal(conv.weight)
            return conv
        self.conv1 = make_conv(12, 128)
        self.conv2 = make_conv(128, 164)
        self.conv3 = make_conv(164, 192)
        self.conv4 = make_conv(192, 192)
        self.conv5 = make_conv(192, 128)
        self.maxpool2 = nn.MaxPool2d(2, ceil_mode=True)
        self.dnn = nn.Linear(128, 128)
        self.output = nn.Linear(128, config['n_labels'])
        self.input_len = 32

    def forward(self, x):
        x = hard_pad2d(x, self.input_len)
        pool_final = nn.MaxPool2d(2, ceil_mode=True) if x.size(2) == 32 else nn.MaxPool2d(3, 1, ceil_mode=True)
        x = self.maxpool2(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.maxpool2(F.relu(self.conv3(x)))
        x = self.maxpool2(F.relu(self.conv4(x)))
        x = pool_final(F.relu(self.conv5(x)))
        x = F.relu(self.dnn(x.view(x.size(0), -1)))
        return F.log_softmax(self.output(x), 1)

class VDPWIModel(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.arch = 'vdpwi'
        self.hidden_dim = config['rnn_hidden_dim']
        self.rnn = nn.LSTM(dim, self.hidden_dim, 1, batch_first=True)
        self.device = config['device']
        if config['classifier'] == 'vdpwi':
            self.classifier_net = VDPWIConvNet(config)
        elif config['classifier'] == 'resnet':
            self.classifier_net = ResNet(config)

    def create_pad_cube(self, sent1, sent2):
        pad_cube = []
        max_len1 = max([len(s.split()) for s in sent1])
        max_len2 = max([len(s.split()) for s in sent2])

        for s1, s2 in zip(sent1, sent2):
            pad1 = (max_len1 - len(s1.split()))
            pad2 = (max_len2 - len(s2.split()))
            pad_mask = np.ones((max_len1, max_len2))
            pad_mask[:len(s1), :len(s2)] = 0
            pad_cube.append(pad_mask)

        pad_cube = np.array(pad_cube)
        return torch.from_numpy(pad_cube).float().to(self.device).unsqueeze(0)

    def compute_sim_cube(self, seq1, seq2):
        def compute_sim(prism1, prism2):
            prism1_len = prism1.norm(dim=3)
            prism2_len = prism2.norm(dim=3)

            dot_prod = torch.matmul(prism1.unsqueeze(3), prism2.unsqueeze(4))
            dot_prod = dot_prod.squeeze(3).squeeze(3)
            cos_dist = dot_prod / (prism1_len * prism2_len + 1E-8)
            l2_dist = ((prism1 - prism2).norm(dim=3))
            return torch.stack([dot_prod, cos_dist, l2_dist], 1)

        def compute_prism(seq1, seq2):
            prism1 = seq1.repeat(seq2.size(1), 1, 1, 1)
            prism2 = seq2.repeat(seq1.size(1), 1, 1, 1)
            prism1 = prism1.permute(1, 2, 0, 3).contiguous()
            prism2 = prism2.permute(1, 0, 2, 3).contiguous()
            return compute_sim(prism1, prism2)

        sim_cube = torch.Tensor(seq1.size(0), 12, seq1.size(1), seq2.size(1))
        sim_cube = sim_cube.to(self.device)
        seq1_f = seq1[:, :, :self.hidden_dim]
        seq1_b = seq1[:, :, self.hidden_dim:]
        seq2_f = seq2[:, :, :self.hidden_dim]
        seq2_b = seq2[:, :, self.hidden_dim:]
        sim_cube[:, 0:3] = compute_prism(seq1, seq2)
        sim_cube[:, 3:6] = compute_prism(seq1_f, seq2_f)
        sim_cube[:, 6:9] = compute_prism(seq1_b, seq2_b)
        sim_cube[:, 9:12] = compute_prism(seq1_f + seq1_b, seq2_f + seq2_b)
        return sim_cube

    def compute_focus_cube(self, sim_cube, pad_cube):
        neg_magic = -10000
        pad_cube = pad_cube.repeat(12, 1, 1, 1)
        pad_cube = pad_cube.permute(1, 0, 2, 3).contiguous()
        sim_cube = neg_magic * pad_cube + sim_cube
        mask = torch.Tensor(*sim_cube.size()).to(self.device)
        mask[:, :, :, :] = 0.1

        def build_mask(index):
            max_mask = sim_cube[:, index].clone()
            for _ in range(min(sim_cube.size(2), sim_cube.size(3))):
                values, indices = torch.max(max_mask.view(sim_cube.size(0), -1), 1)
                row_indices = indices / sim_cube.size(3)
                col_indices = indices % sim_cube.size(3)
                row_indices = row_indices.unsqueeze(1)
                col_indices = col_indices.unsqueeze(1).unsqueeze(1)
                for i, (row_i, col_i, val) in enumerate(zip(row_indices, col_indices, values)):
                    if val < neg_magic / 2:
                        continue
                    mask[i, :, row_i, col_i] = 1
                    max_mask[i, row_i, :] = neg_magic
                    max_mask[i, :, col_i] = neg_magic
        build_mask(9)
        build_mask(10)
        focus_cube = mask * sim_cube * (1 - pad_cube)
        return focus_cube

    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None):
        pad_cube = self.create_pad_cube(raw_sent1, raw_sent2)
        sent1 = sent1.permute(0, 2, 1).contiguous()
        sent2 = sent2.permute(0, 2, 1).contiguous()
        seq1f, _ = self.rnn(sent1)
        seq2f, _ = self.rnn(sent2)
        seq1b, _ = self.rnn(torch.cat(sent1.split(1, 1)[::-1], 1))
        seq2b, _ = self.rnn(torch.cat(sent2.split(1, 1)[::-1], 1))
        seq1 = torch.cat([seq1f, seq1b], 2)
        seq2 = torch.cat([seq2f, seq2b], 2)
        sim_cube = self.compute_sim_cube(seq1, seq2)
        truncate = self.classifier_net.input_len
        sim_cube = sim_cube[:, :, :pad_cube.size(2), :pad_cube.size(3)].contiguous()
        if truncate is not None:
            sim_cube = sim_cube[:, :, :truncate, :truncate].contiguous()
            pad_cube = pad_cube[:, :, :sim_cube.size(2), :sim_cube.size(3)].contiguous()
        focus_cube = self.compute_focus_cube(sim_cube, pad_cube)
        log_prob = self.classifier_net(focus_cube)
        return log_prob
