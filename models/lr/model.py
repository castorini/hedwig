import torch
import torch.nn as nn


class LogisticRegression(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.vocab_size, config.num_labels)

    def forward(self, x, **kwargs):
        x = torch.squeeze(x)  # (batch, vocab_size)
        x = self.dropout(x)
        logit = self.fc1(x)  # (batch, target_size)
        return logit
