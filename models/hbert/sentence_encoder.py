from torch import nn
from transformers import BertPreTrainedModel, BertModel


class BertSentenceEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config, num_labels=config.num_labels)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        pooled_output = self.bert(input_ids, attention_mask, token_type_ids)[1]
        pooled_output = self.dropout(pooled_output)
        return pooled_output
