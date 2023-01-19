import torch
from torch import nn
from transformers import AutoModel

MODEL_NAME = "cointegrated/rubert-tiny2"


class ComplexModel(nn.Module):
    def __init__(self, bert_model_name):
        super(ComplexModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)

        self.manipulation_dense = nn.Linear(self.bert.config.hidden_size, 37)
        self.RE_bilin1 = nn.Bilinear(
            self.bert.config.hidden_size,
            self.bert.config.hidden_size,
            512
        )
        self.RE_relu = nn.ReLU()
        self.RE_lin1 = nn.Linear(512, 512)
        self.RE_sigmoid = nn.Sigmoid()

    # maniputalion detection part
    def _manipulation_forward(self, bert_output):
        return self.manipulation_dense(bert_output)

    # relation extraction part
    def _connections_forward(self, bert_output):
        h = self.RE_bilin1(bert_output, bert_output)
        h = self.RE_relu(h)
        h = self.RE_lin1(h)
        connection_output = self.RE_sigmoid(h)
        return connection_output

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        bert_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, return_dict=False)
        manipulation_output = self._manipulation_forward(bert_output)
        connection_output = self._connections_forward(bert_output)

        return manipulation_output, connection_output
