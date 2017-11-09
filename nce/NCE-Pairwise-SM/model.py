import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseConv(nn.Module):
    """docstring for PairwiseConv"""
    def __init__(self, model):
        super(PairwiseConv, self).__init__()
        self.convModel = model
        self.dropout = nn.Dropout(self.convModel.dropout)
        self.linearLayer = nn.Linear(model.n_hidden, 1)
        self.posModel = self.convModel
        # share or copy ??
        # https://discuss.pytorch.org/t/copying-nn-modules-without-shared-memory/113
        # self.negModel = copy.deepcopy(self.posModel)
        self.negModel = self.convModel

    def forward(self, input):
        pos = self.posModel(input[0])
        neg = self.negModel(input[1])
        pos = self.dropout(pos)
        neg = self.dropout(neg)
        pos = self.linearLayer(pos)
        neg = self.linearLayer(neg)
        combine = torch.cat([pos, neg], 1)
        return combine

class SmPlusPlus(nn.Module):
    def __init__(self, config):
        super(SmPlusPlus, self).__init__()
        output_channel = config.output_channel
        questions_num = config.questions_num
        answers_num = config.answers_num
        words_dim = config.words_dim
        filter_width = config.filter_width
        self.mode = config.mode
        self.dropout = config.dropout

        n_classes = config.target_class
        ext_feats_size = config.ext_feats_size

        if self.mode == 'multichannel':
            input_channel = 2
        else:
            input_channel = 1

        self.question_embed = nn.Embedding(questions_num, words_dim)
        self.answer_embed = nn.Embedding(answers_num, words_dim)
        self.static_question_embed = nn.Embedding(questions_num, words_dim)
        self.nonstatic_question_embed = nn.Embedding(questions_num, words_dim)
        self.static_answer_embed = nn.Embedding(answers_num, words_dim)
        self.nonstatic_answer_embed = nn.Embedding(answers_num, words_dim)
        self.static_question_embed.weight.requires_grad = False
        self.static_answer_embed.weight.requires_grad = False

        self.conv_q = nn.Conv2d(input_channel, output_channel, (filter_width, words_dim), padding=(filter_width - 1, 0))
        self.conv_a = nn.Conv2d(input_channel, output_channel, (filter_width, words_dim), padding=(filter_width - 1, 0))

        self.n_hidden = 2 * output_channel + ext_feats_size

        self.combined_feature_vector = nn.Linear(self.n_hidden, self.n_hidden)
        self.hidden = nn.Linear(self.n_hidden, n_classes)

    def forward(self, x):
        x_question = x.sentence_1
        x_answer = x.sentence_2
        x_ext = x.ext_feats

        if self.mode == 'rand':
            question = self.question_embed(x_question).unsqueeze(1)
            answer = self.answer_embed(x_answer).unsqueeze(1) # (batch, sent_len, embed_dim)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        # actual SM model mode (Severyn & Moschitti, 2015)
        elif self.mode == 'static':
            question = self.static_question_embed(x_question).unsqueeze(1)
            answer = self.static_answer_embed(x_answer).unsqueeze(1) # (batch, sent_len, embed_dim)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        elif self.mode == 'non-static':
            question = self.nonstatic_question_embed(x_question).unsqueeze(1)
            answer = self.nonstatic_answer_embed(x_answer).unsqueeze(1) # (batch, sent_len, embed_dim)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        elif self.mode == 'multichannel':
            question_static = self.static_question_embed(x_question)
            answer_static = self.static_answer_embed(x_answer)
            question_nonstatic = self.nonstatic_question_embed(x_question)
            answer_nonstatic = self.nonstatic_answer_embed(x_answer)
            question = torch.stack([question_static, question_nonstatic], dim=1)
            answer = torch.stack([answer_static, answer_nonstatic], dim=1)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        else:
            print("Unsupported Mode")
            exit()

        x.append(x_ext)
        x = torch.cat(x, 1)
        x = F.tanh(self.combined_feature_vector(x))

        return x