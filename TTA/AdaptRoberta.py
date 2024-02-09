from utils.config_classes import ModelArguments, DataTrainingArguments
from transformers.modeling_outputs import (SequenceClassifierOutput,
    TokenClassifierOutput,)
from transformers.modeling_roberta import *

from transformers import RobertaModel, RobertaConfig
import torch
import torch.nn as nn
from transformers.modeling_outputs import (
    SequenceClassifierOutput, 
    TokenClassifierOutput, 
    QuestionAnsweringModelOutput
)
import transformers

import logging
import os
import random
import sys
import warnings
from typing import List, Optional
import datasets
import evaluate
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
logger = logging.getLogger(__name__)

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

class AdaptedRobertaForSequenceClassification(RobertaModel):
    """
    Custom Roberta model for handling multiple tasks: classification, NER, QA.
    Inherits from RobertaModel and adds task-specific heads and an adapter.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config) 
        
        #self.adapter = Adapter(config)  #this is only one adapter.
        # Add adapters. 2 adapters
        self.adapter1 = nn.ModuleList([Adapter(config) for _ in range(config.num_hidden_layers // 2)])
        self.adapter2 = nn.ModuleList([Adapter(config) for _ in range(config.num_hidden_layers // 2)])

        # Add side block output head.
        self.adapter1_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.adapter2_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):#-> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # Get the outputs of each layer
        '''outputs = super().forward(
            input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            output_hidden_states=True
        )'''
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        ) # self.roberta is a RobertaModel class, the output is class BaseModelOutputWithPoolingAndCrossAttentions
        #contains:last_hidden_state;pooler_output;hidden_states;past_key_values;attentions;cross_attentions

        sequence_output = outputs[0] # last_hidden_state, the sequence output
        
        #1logits = self.classifier(sequence_output)
        layer_outputs = outputs[2] # hidden_states

        adapter1_states = layer_outputs[0]
        adapter2_states = layer_outputs[0]
        # Side block forward propagation.
        for i in range(1, len(layer_outputs), 2):
            adapter1_states = self.adapter1[i // 2](adapter1_states + layer_outputs[i] + layer_outputs[i + 1])

        for i in range(0, len(layer_outputs)-1, 2):
            adapter2_states = self.adapter1[i // 2](adapter2_states + layer_outputs[i] + layer_outputs[i + 1])
   
        pooled_output = adapted_sequence[:, 0, :]  # [CLS] token for classification tasks
        logits = self.classification_head(pooled_output)
        return SequenceClassifierOutput(logits=logits)
        

class AdaptedRobertaForTokenClassification(RobertaModel):
    """
    Custom Roberta model for handling multiple tasks: classification, NER, QA.
    Inherits from RobertaModel and adds task-specific heads and an adapter.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # Task-specific heads

        ner_classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(ner_classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        #self.qa_head = nn.Linear(config.hidden_size, config.num_labels)  #this cannot be used for generative QA(need to use model like T5, bloom)
        # For start and end logits in QA config.num_labels=2，also can be self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        
        #self.adapter = Adapter(config)  #this is only one adapter.
        # Add adapters. 2 adapters
        self.adapter1 = nn.ModuleList([Adapter(config) for _ in range(config.num_hidden_layers // 2)])
        self.adapter2 = nn.ModuleList([Adapter(config) for _ in range(config.num_hidden_layers // 2)])

        # Add side block output head.
        self.adapter1_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.adapter2_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):#-> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # Get the outputs of each layer
        '''outputs = super().forward(
            input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            output_hidden_states=True
        )'''
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        ) # self.roberta is a RobertaModel class, the output is class BaseModelOutputWithPoolingAndCrossAttentions
        #contains:last_hidden_state;pooler_output;hidden_states;past_key_values;attentions;cross_attentions

        sequence_output = outputs[0] # last_hidden_state, the sequence output
        sequence_output = self.dropout(sequence_output)
        #1logits = self.classifier(sequence_output)
        layer_outputs = outputs[2] # hidden_states

        adapter1_states = layer_outputs[0]
        adapter2_states = layer_outputs[0]
        # Side block forward propagation.
        for i in range(1, len(layer_outputs), 2):
            adapter1_states = self.adapter1[i // 2](adapter1_states + layer_outputs[i] + layer_outputs[i + 1])

        for i in range(0, len(layer_outputs)-1, 2):
            adapter2_states = self.adapter1[i // 2](adapter2_states + layer_outputs[i] + layer_outputs[i + 1])

        logits = self.ner_head(adapted_sequence)
        return TokenClassifierOutput(logits=logits)
      
# Example usage
config = RobertaConfig.from_pretrained("roberta-base")
config.num_labels = 2  # For binary classification
config.num_labels_ner = 10  # For NER with 10 different entities
model = CustomRobertaForMultipleTasks(config)

# Example inputs
input_ids = torch.tensor([[0, 1, 2, 3, 4]])  # Example input IDs
attention_mask = torch.tensor([[1, 1, 1, 1, 1]])  # Example attention mask

# Get outputs for a specific task
classification_output = model(input_ids, attention_mask, task="classification")
ner_output = model(input_ids, attention_mask, task="ner")
qa_output = model(input_ids, attention_mask, task="qa")

#-----------------------------------------------------------------------------------------------------------------------