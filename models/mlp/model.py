import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        target_class = config.target_class
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(dataset.VOCAB_SIZE, target_class)

    def forward(self, x, **kwargs):
        x = torch.squeeze(x)  # (batch, vocab_size)
        x = self.dropout(x)
        logit = self.fc1(x)  # (batch, target_size)
        return logit
