import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, RobertaConfig, RobertaModel


class RobertaForWsd(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    # override forward
    def forward(self, input_ids, attention_mask, token_type_ids, instance_mask, instance_lens, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        instance_mask = instance_mask.unsqueeze(dim=2).repeat(1, 1, self.hidden_size)
        instance_lens = instance_lens.unsqueeze(dim=1).repeat(1, self.hidden_size)
        preds = (outputs[0] * instance_mask).sum(dim=1) / instance_lens
        preds = torch.cat((outputs[1], preds), dim=1)
        preds = self.dropout(preds)
        logits = self.classifier(preds)
        probs = self.softmax(logits)
        if labels is not None:
            return (self.criterion(logits, labels), probs[:, 1].contiguous())
        else:
            return (probs[:, 1].contiguous(),)


def get_wsd_model(config_name):
    model_dict = {
        'RobertaConfig': RobertaForWsd
    }
    return model_dict[config_name]
