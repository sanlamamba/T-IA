import torch.nn as nn
from helpers.model_helper import BertForSequenceClassificationBase
from transformers import AutoModel


class BertForSequenceClassification(BertForSequenceClassificationBase):
    def __init__(
        self, 
        n_depart, 
        n_arrival, 
        bert_model=AutoModel.from_pretrained("dbmdz/bert-base-french-europeana-cased")
    ):
        super(BertForSequenceClassification, self).__init__()
        self.bert = bert_model
        self.drop = nn.Dropout(p=0.3)
        self.out_depart = nn.Linear(self.bert.config.hidden_size, n_depart)
        self.out_arrival = nn.Linear(self.bert.config.hidden_size, n_arrival)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]

        output_depart = self.out_depart(self.drop(pooled_output))
        output_arrival = self.out_arrival(self.drop(pooled_output))

        return output_depart, output_arrival

    def freeze_arrival_layer(self):
        for param in self.out_arrival.parameters():
            param.requires_grad = False

    def unfreeze_arrival_layer(self):
        for param in self.out_arrival.parameters():
            param.requires_grad = True

    def freeze_depart_layer(self):
        for param in self.out_depart.parameters():
            param.requires_grad = False

    def unfreeze_depart_layer(self):
        for param in self.out_depart.parameters():
            param.requires_grad = True