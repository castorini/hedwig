import torch
import torch.nn as nn

from models.han.sent_level_rnn import SentLevelRNN
from models.han.word_level_rnn import WordLevelRNN


class HAN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.word_attention_rnn = WordLevelRNN(config)
        self.sentence_attention_rnn = SentLevelRNN(config)

    def forward(self, x,  **kwargs):
        x = x.permute(1, 2, 0) # Expected : # sentences, # words, batch size
        num_sentences = x.size(0)
        word_attentions = None
        for i in range(num_sentences):
            word_attn = self.word_attention_rnn(x[i, :, :])
            if word_attentions is None:
                word_attentions = word_attn
            else:
                word_attentions = torch.cat((word_attentions, word_attn), 0)
        return self.sentence_attention_rnn(word_attentions)

