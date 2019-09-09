import torch
import torch.nn as nn

import torch.nn.functional as F


class FastText(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        target_class = config.target_class
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode

        if config.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif config.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif config.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            print("Unsupported Mode")
            exit()

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(words_dim, target_class)

    def forward(self, x, **kwargs):
        if self.mode == 'rand':
            x = self.embed(x)  # (batch, sent_len, embed_dim)
        elif self.mode == 'static':
            x = self.static_embed(x)  # (batch, sent_len, embed_dim)
        elif self.mode == 'non-static':
            x = self.non_static_embed(x)  # (batch, sent_len, embed_dim)

        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze(1)  # (batch, embed_dim)

        logit = self.fc1(x)  # (batch, target_size)
        return logit


