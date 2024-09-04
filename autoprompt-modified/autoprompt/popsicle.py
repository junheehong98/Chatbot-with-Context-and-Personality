"""
Frozen model with a linear topping...I'm really sleepy...
"""
import logging

import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    AutoConfig,
    BertConfig,
    BertForSequenceClassification,
    PretrainedConfig,
    RobertaConfig,
    RobertaForSequenceClassification
)


logger = logging.getLogger(__name__)



class Bertsicle(BertForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        num_classes = 3  # 각 특성에 대해 예측할 점수의 개수 (1점, 3점, 5점)
        num_features = 5  # 예측할 감정 특성의 개수
        self.classifier = torch.nn.Linear(config.hidden_size, num_classes * num_features)  # 출력 크기: 3 x 5

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[0]
        pooled_output = pooled_output[:,1:,:] #eliminating CLS token
        pooled_output = torch.mean(pooled_output, dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output).view(-1, 5, 3)  # 5개의 특성, 각 특성당 3개의 점수 예측

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class Robertasicle(RobertaForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        num_classes = 3  # 각 특성에 대해 예측할 점수의 개수 (1점, 3점, 5점)
        num_features = 5  # 예측할 감정 특성의 개수
        self.classifier = torch.nn.Linear(config.hidden_size, num_classes * num_features)  # 출력 크기: 3 x 5

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
       
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, 1:, :]  # eliminating <s> token
        pooled_sequence_output = torch.mean(sequence_output, dim=1, keepdim=True)
        logits = self.classifier(pooled_sequence_output)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


MODEL_MAPPING = {
        RobertaConfig: Robertasicle,
        BertConfig: Bertsicle
}


class AutoPopsicle:
    def __init__(self):
        raise EnvironmentError('You done goofed. Use `.from_pretrained()` or something.')

    @classmethod
    def from_config(cls, config):
        for config_class, model_class in MODEL_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError('We do not support this config.')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_MAPPING.items():
            if isinstance(config, config_class):
                logger.info(f'Config class: {config_class}')
                logger.info(f'Model class: {model_class}')
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        raise ValueError('We do not support "{pretrained_model_name_or_path}".')
