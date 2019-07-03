import torch
from torch import nn
import torch.nn.functional as F

from models.hbert.sentence_encoder import BertSentenceEncoder


class HierarchicalBert(nn.Module):

    def __init__(self, args, cache_dir):
        super().__init__()
        self.args =args
        ks = 3
        input_channels = 1

        self.sentence_encoder = BertSentenceEncoder.from_pretrained(
            args.model,
            cache_dir=cache_dir,
            num_labels=args.num_labels)

        self.conv1 = nn.Conv2d(input_channels,
                               args.conv_output_channels,
                               (3, self.sentence_encoder.config.hidden_size),
                               padding=(2, 0))
        self.conv2 = nn.Conv2d(input_channels,
                               args.conv_output_channels,
                               (4, self.sentence_encoder.config.hidden_size),
                               padding=(3, 0))
        self.conv3 = nn.Conv2d(input_channels,
                               args.conv_output_channels,
                               (5, self.sentence_encoder.config.hidden_size),
                               padding=(4, 0))

        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(ks * args.conv_output_channels, args.num_labels)

    def forward(self, input_ids, segment_ids=None, input_mask=None):
        """
        a batch is a tensor of shape [batch_size, #file_in_commit, #line_in_file]
        and each element is a line, i.e., a bert_batch,
        which consists of input_ids, input_mask, segment_ids, label_ids
        """
        input_ids = input_ids.permute(1, 0, 2)  # (sentences, batch_size, words)
        segment_ids = segment_ids.permute(1, 0, 2)
        input_mask = input_mask.permute(1, 0, 2)

        x_encoded = []
        for i0 in range(len(input_ids)):
            x_encoded.append(self.sentence_encoder(input_ids[i0], input_mask[i0], segment_ids[i0]))

        x = torch.stack(x_encoded)  # (sentences, batch_size, hidden_size)
        x = x.permute(1, 0, 2)  # (batch_size, sentences, hidden_size)
        x = x.unsqueeze(1)  # (batch_size, input_channels, sentences, hidden_size)

        if self.args.batchnorm:
            x = [F.relu(self.batchnorm1(self.conv1(x))).squeeze(3),
                 F.relu(self.batchnorm2(self.conv2(x))).squeeze(3),
                 F.relu(self.batchnorm3(self.conv3(x))).squeeze(3)]
        else:
            x = [F.relu(self.conv1(x)).squeeze(3),
                 F.relu(self.conv2(x)).squeeze(3),
                 F.relu(self.conv3(x)).squeeze(3)]

        if self.args.dynamic_pool:
            x = [self.dynamic_pool(i).squeeze(2) for i in x]  # (batch, output_channels) * ks
            x = torch.cat(x, 1)  # (batch, output_channels * ks)
            x = x.view(-1, self.filter_widths * self.output_channel * self.dynamic_pool_length)
        else:
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (batch, output_channels, num_sentences) * ks
            x = torch.cat(x, 1)  # (batch, channel_output * ks)

        x = self.dropout(x)
        logit = self.fc1(x)  # (batch, num_labels)

        return logit
