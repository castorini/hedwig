import torch
import torch.nn as nn
import torch.nn.functional as F


class XmlCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        self.output_channel = config.output_channel
        target_class = config.target_class
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode
        self.num_bottleneck_hidden = config.num_bottleneck_hidden
        self.dynamic_pool_length = config.dynamic_pool_length
        self.ks = 3 # There are three conv nets here

        input_channel = 1
        if config.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif config.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif config.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        elif config.mode == 'multichannel':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
            input_channel = 2
        else:
            print("Unsupported Mode")
            exit()

        ## Different filter sizes in xml_cnn than kim_cnn
        self.conv1 = nn.Conv2d(input_channel, self.output_channel, (2, words_dim), padding=(1,0))
        self.conv2 = nn.Conv2d(input_channel, self.output_channel, (4, words_dim), padding=(3,0))
        self.conv3 = nn.Conv2d(input_channel, self.output_channel, (8, words_dim), padding=(7,0))

        self.dropout = nn.Dropout(config.dropout)
        self.bottleneck = nn.Linear(self.ks * self.output_channel * self.dynamic_pool_length, self.num_bottleneck_hidden)
        self.fc1 = nn.Linear(self.num_bottleneck_hidden, target_class)

        self.pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length) #Adaptive pooling

    def forward(self, x, **kwargs):
        if self.mode == 'rand':
            word_input = self.embed(x) # (batch, sent_len, embed_dim)
            x = word_input.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'static':
            static_input = self.static_embed(x)
            x = static_input.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'non-static':
            non_static_input = self.non_static_embed(x)
            x = non_static_input.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'multichannel':
            non_static_input = self.non_static_embed(x)
            static_input = self.static_embed(x)
            x = torch.stack([non_static_input, static_input], dim=1) # (batch, channel_input=2, sent_len, embed_dim)
        else:
            print("Unsupported Mode")
            exit()
        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        x = [self.pool(i).squeeze(2) for i in x]

        # (batch, channel_output) * ks
        x = torch.cat(x, 1) # (batch, channel_output * ks)
        x = F.relu(self.bottleneck(x.view(-1, self.ks * self.output_channel * self.dynamic_pool_length)))
        x = self.dropout(x)
        logit = self.fc1(x) # (batch, target_size)
        return logit
