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
        num_classes = 2  #  각 특성에 대해 예측할 점수의 개수 (0점, 1점)
        num_features = 5  # 예측할 감정 특성의 개수
        hidden_size = config.hidden_size  # BERT의 hidden size


        self.bert.eval()
        # BERT 파라미터 freeze
        for param in self.bert.parameters():
            param.requires_grad = False       
        
        # 5개의 Dense Layer 추가
        self.dense1 = torch.nn.Linear(hidden_size, hidden_size)
        self.dense2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.dense3 = torch.nn.Linear(hidden_size // 2, hidden_size // 4)
        self.dense4 = torch.nn.Linear(hidden_size // 4, hidden_size // 8)
        self.dense5 = torch.nn.Linear(hidden_size // 8, num_classes * num_features)
        self.dropout = torch.nn.Dropout(0.3)


        # self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        # self.bn2 = torch.nn.BatchNorm1d(hidden_size // 2)
        # self.bn3 = torch.nn.BatchNorm1d(hidden_size // 4)
        # self.bn4 = torch.nn.BatchNorm1d(hidden_size // 8)

        # Layer Normalization으로 변경
        self.ln1 = torch.nn.LayerNorm(hidden_size)
        self.ln2 = torch.nn.LayerNorm(hidden_size // 2)
        self.ln3 = torch.nn.LayerNorm(hidden_size // 4)
        self.ln4 = torch.nn.LayerNorm(hidden_size // 8)









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
        
        
        #  # 두 개의 dense layer를 통과
        # x = torch.relu(self.bn1(self.dense1(pooled_output)))
        # x = torch.relu(self.bn2(self.dense2(x)))
        # x = torch.relu(self.bn3(self.dense3(x)))
        # x = torch.relu(self.bn4(self.dense4(x)))
        x = torch.relu(self.ln1(self.dense1(pooled_output)))
        x = torch.relu(self.ln2(self.dense2(x)))
        x = torch.relu(self.ln3(self.dense3(x)))
        x = torch.relu(self.ln4(self.dense4(x)))

        logits = self.dense5(x)


        logits = logits.view(-1, 5, 2)  # 5개의 특성, 각 특성당 3개의 점수 예측
        
        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        outputs = (logits,)  # Hidden states 및 Attention은 생략

        
        if labels is not None:
            '''
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
            outputs = (loss,) + outputs
            '''
            # 손실 계산을 여기서 제거합니다.
            pass  # 또는 아무 작업도 하지 않음

        return outputs  # (loss), logits, (hidden_states), (attentions)


class Robertasicle(RobertaForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        num_classes = 2  # 각 특성에 대해 예측할 점수의 개수 (1점, 3점, 5점)
        num_features = 5  # 예측할 감정 특성의 개수
        
        hidden_size = config.hidden_size  # RoBERTa의 hidden size

        # RoBERTa 파라미터 freeze
        for param in self.roberta.parameters():
            param.requires_grad = False

        # 두 개의 dense layer 추가
        self.dense1 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.dense2 = torch.nn.Linear(hidden_size // 2, num_classes * num_features)
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
        # 두 개의 dense layer를 통과
        x = torch.relu(self.dense1(pooled_sequence_output))
        logits = self.dense2(x)

        logits = logits.view(-1, 5, 2)  # 5개의 특성, 각 특성당 3개의 점수 예측
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
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
