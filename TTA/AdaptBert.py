from utils.config_classes import ModelArguments, DataTrainingArguments
from transformers.modeling_outputs import (SequenceClassifierOutput,
    TokenClassifierOutput,)
from transformers.modeling_bert import *
from transformers.modeling_bert import (BertForSequenceClassification,BertForTokenClassification)

from transformers import BertModel, BertConfig
import torch
import torch.nn as nn
from transformers.modeling_outputs import (
    SequenceClassifierOutput, 
    TokenClassifierOutput, 
    QuestionAnsweringModelOutput
)

from typing import List, Optional

import numpy as np


class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.down = nn.Linear(config.hidden_size, config.hidden_size) # (in_feature, out_feature) didn't change here
        self.act = torch.nn.GELU()
        self.up = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.LayerNorm(x + self.dropout(self.up(self.act(self.down(x)))))

def entropy(tensor):
    return -(tensor.softmax(dim=-1) * tensor.log_softmax(dim=-1)).sum(dim=-1)

def kl(source, target):
    return torch.sum(target.softmax(dim=-1) * (target.log_softmax(dim=-1) - source.log_softmax(dim=-1)), dim=-1)

class AdaptedBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.adapter = Adapter(config)  #this is only one adapter.
        # Add adapters. 2 adapters
        self.adapter1 = nn.ModuleList([Adapter(config) for _ in range(config.num_hidden_layers // 2)])
        self.adapter2 = nn.ModuleList([Adapter(config) for _ in range(config.num_hidden_layers // 2)])

        # Add side block output head.
        self.adapter1_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.adapter2_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None

        layer_outputs = outputs[2]
        # Initialize the side block state as the output of the embedding.
        adapter1_states = layer_outputs[0]
        # Side block forward propagation.
        for i in range(1, len(layer_outputs), 2):
            adapter1_states = self.adapter1[i // 2](adapter1_states + layer_outputs[i] + layer_outputs[i + 1])
        adapter1_logits = self.adapter1_outputs(adapter1_states) #size=[batch_size, num_labels]?

        for i in range(0, len(layer_outputs)-1, 2):
            adapter1_states = self.adapter1[i // 2](adapter1_states + layer_outputs[i] + layer_outputs[i + 1])
        adapter2_logits = self.adapter2_outputs(adapter2_states)
        ada_logits=
        ada_loss==entropy(ada_logits).mean()
        adapter1kl=kl()


        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



#-----------------------------------------------------------------------------------------------------------------------