from config_classes import ModelArguments, DataTrainingArguments
from transformers import RobertaModel, RobertaConfig
import torch
import torch.nn as nn
from transformers.modeling_outputs import (
    SequenceClassifierOutput, 
    TokenClassifierOutput, 
    QuestionAnsweringModelOutput
)
import transformers
from transformers.modeling_roberta import *
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

class AdaptedRobertaForMultipleTasks(RobertaModel):
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
        if config.task_name == "classification":
            self.classifier = RobertaClassificationHead(config) 
        elif config.task_name == "ner":
            ner_classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
            self.dropout = nn.Dropout(ner_classifier_dropout)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        #self.qa_head = nn.Linear(config.hidden_size, config.num_labels)  #this cannot be used for generative QA(need to use model like T5, bloom)
        # For start and end logits in QA config.num_labels=2，also can be self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        
        #self.adapter = Adapter(config)  #this is only one adapter.
        # Add adapters.
        self.adapter = nn.ModuleList([Adapter(config) for _ in range(config.num_hidden_layers // 2)])

        # Add side block output head.
        self.adapter_outputs = nn.Linear(config.hidden_size, config.num_labels)
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
        if config.task_name == "ner":
            sequence_output = self.dropout(sequence_output)
        #1logits = self.classifier(sequence_output)
        layer_outputs = outputs[2] # hidden_states


        # Sum the input and output of each Transformer block
        transformer_blocks = self.roberta.encoder.layer
        hidden_states = outputs.hidden_states
        adapted_sequence = hidden_states[0]  # Initial embeddings as the first input

        for i, layer_module in enumerate(transformer_blocks):
            layer_outputs = layer_module(adapted_sequence, attention_mask)
            layer_output = layer_outputs[0]
            adapted_sequence = hidden_states[i] + layer_output

        # Apply adapter on the summed output
        adapted_sequence = self.adapter(adapted_sequence)

        if config.task_name == "classification":
            pooled_output = adapted_sequence[:, 0, :]  # [CLS] token for classification tasks
            logits = self.classification_head(pooled_output)
            return SequenceClassifierOutput(logits=logits)
        elif config.task_name == "ner":
            logits = self.ner_head(adapted_sequence)
            return TokenClassifierOutput(logits=logits)
        elif config.task_name == "qa":
            logits = self.qa_head(adapted_sequence)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            return QuestionAnsweringModelOutput(start_logits=start_logits, end_logits=end_logits)
        else:
            raise ValueError('need to specify config.task_name')


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

def main():
    # See all possible arguments in src/transformers/training_args.py or by passing the --help flag to this script.
    ## -------------------------------Parsing and Logging--------------------------------------------------------------
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    #---------------------------------------------------------------------------------------------------------------------
            
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files, or specify a dataset name
    # to load from huggingface/datasets. In ether case, you can specify a the key of the column(s) containing the text and
    # the key of the column containing the label. If multiple columns are specified for the text, they will be joined togather
    # for the actual text value.
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
        # Try print some info about the dataset
        logger.info(f"Dataset loaded: {raw_datasets}")
        logger.info(raw_datasets)
    else:
        data_files = {}            
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir,token=model_args.token)

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")
    
    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        #finetuning_task="text-classification", 
        task_name=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    #QA: AutoModelForSeq2SeqLM.from_pretrained()  ner: AutoModelForTokenClassification

    ## data    
    if data_args.task_name == 'classification':
        if data_args.remove_splits is not None:
            for split in data_args.remove_splits.split(","):
                logger.info(f"removing split {split}")
                raw_datasets.pop(split)

        if data_args.train_split_name is not None:
            logger.info(f"using {data_args.validation_split_name} as validation set")
            raw_datasets["train"] = raw_datasets[data_args.train_split_name]
            raw_datasets.pop(data_args.train_split_name)

        if data_args.validation_split_name is not None:
            logger.info(f"using {data_args.validation_split_name} as validation set")
            raw_datasets["validation"] = raw_datasets[data_args.validation_split_name]
            raw_datasets.pop(data_args.validation_split_name)

        if data_args.test_split_name is not None:
            logger.info(f"using {data_args.test_split_name} as test set")
            raw_datasets["test"] = raw_datasets[data_args.test_split_name]
            raw_datasets.pop(data_args.test_split_name)

        if data_args.remove_columns is not None:
            for split in raw_datasets.keys():
                for column in data_args.remove_columns.split(","):
                    logger.info(f"removing column {column} from split {split}")
                    raw_datasets[split].remove_columns(column)

        if data_args.label_column_name is not None and data_args.label_column_name != "label":
            for key in raw_datasets.keys():
                raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = (
            raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
            if data_args.do_regression is None
            else data_args.do_regression
        )

        is_multi_label = False
        if is_regression:
            label_list = None
            num_labels = 1
            # regession requires float as label type, let's cast it if needed
            for split in raw_datasets.keys():
                if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
                    logger.warning(
                        f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}"
                    )
                    features = raw_datasets[split].features
                    features.update({"label": Value("float32")})
                    try:
                        raw_datasets[split] = raw_datasets[split].cast(features)
                    except TypeError as error:
                        logger.error(
                            f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                        )
                        raise error

        else:  # classification
            if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
                is_multi_label = True
                logger.info("Label type is list, doing multi-label classification")
            # Trying to find the number of labels in a multi-label classification task
            # We have to deal with common cases that labels appear in the training set but not in the validation/test set.
            # So we build the label list from the union of labels in train/val/test.
            label_list = get_label_list(raw_datasets, split="train")
            for split in ["validation", "test"]:
                if split in raw_datasets:
                    val_or_test_labels = get_label_list(raw_datasets, split=split)
                    diff = set(val_or_test_labels).difference(set(label_list))
                    if len(diff) > 0:
                        # add the labels that appear in val/test but not in train, throw a warning
                        logger.warning(
                            f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                        )
                        label_list += list(diff)
            # if label is -1, we throw a warning and remove it from the label list
            for label in label_list:
                if label == -1:
                    logger.warning("Label -1 found in label list, removing it.")
                    label_list.remove(label)

            label_list.sort()
            num_labels = len(label_list)
            if num_labels <= 1:
                raise ValueError("You need more than one label to do classification.")
        if is_regression:
            config.problem_type = "regression"
            logger.info("setting problem type to regression")
        elif is_multi_label:
            config.problem_type = "multi_label_classification"
            logger.info("setting problem type to multi label classification")
        else:
            config.problem_type = "single_label_classification"
            logger.info("setting problem type to single label classification")

        # Padding strategy
        if data_args.pad_to_max_length:
            padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            padding = False    
    ##############----------NER-----------#####################
    if data_args.task_name == 'ner':
        if training_args.do_train:
            column_names = raw_datasets["train"].column_names
            features = raw_datasets["train"].features
        else:
            column_names = raw_datasets["validation"].column_names
            features = raw_datasets["validation"].features

        # if text_column_name and label_column_name are not specified, use tokens and ner_tags.
        if data_args.text_column_name is not None:
            text_column_name = data_args.text_column_name
        elif "tokens" in column_names:
            text_column_name = "tokens"
        else:
            text_column_name = column_names[0]

        if data_args.label_column_name is not None:
            label_column_name = data_args.label_column_name
        elif f"{data_args.task_name}_tags" in column_names:
            label_column_name = f"{data_args.task_name}_tags"
        else:
            label_column_name = column_names[1]

        # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
        # unique labels.
        def get_label_list(labels):
            unique_labels = set()
            for label in labels:
                unique_labels = unique_labels | set(label)
            label_list = list(unique_labels)
            label_list.sort()
            return label_list

        # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
        # Otherwise, we have to get the list of labels manually.
        labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
        if labels_are_int:
            label_list = features[label_column_name].feature.names
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            label_list = get_label_list(raw_datasets["train"][label_column_name])
            label_to_id = {l: i for i, l in enumerate(label_list)}

        num_labels = len(label_list)
    #---------------------------------------------------------------------
         
    # for training ,we will update the config with label infos,
    # if do_train is not set, we will use the label infos in the config
    if training_args.do_train and not is_regression:  # classification, training
        label_to_id = {v: i for i, v in enumerate(label_list)}
        # update config with label infos
        if model.config.label2id != label_to_id:
            logger.warning(
                "The label2id key in the model config.json is not equal to the label2id key of this "
                "run. You can ignore this if you are doing finetuning."
            )
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif not is_regression:  # classification, but not training
        logger.info("using label infos in the model config")
        logger.info("label2id: {}".format(model.config.label2id))
        label_to_id = model.config.label2id
    else:  # regression
        label_to_id = None

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def multi_labels_to_ids(labels: List[str]) -> List[float]:
        ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
        for label in labels:
            ids[label_to_id[label]] = 1.0
        return ids

    def preprocess_function(examples):
        if data_args.text_column_names is not None:
            text_column_names = data_args.text_column_names.split(",")
            # join together text columns into "sentence" column
            examples["sentence"] = examples[text_column_names[0]]
            for column in text_column_names[1:]:
                for i in range(len(examples[column])):
                    examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]
        # Tokenize the texts
        result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)
        if label_to_id is not None and "label" in examples:
            if is_multi_label:
                result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
            else:
                result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
        return result

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
            else:
                logger.warning("Validation dataset not found. Falling back to test dataset for validation.")
                eval_dataset = raw_datasets["test"]
        else:
            eval_dataset = raw_datasets["validation"]

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        # remove label column if it exists
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    '''# Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")'''

    if data_args.metric_name is not None:
        metric = (
            evaluate.load(data_args.metric_name, config_name="multilabel")
            if is_multi_label
            else evaluate.load(data_args.metric_name)
        )
        logger.info(f"Using metric {data_args.metric_name} for evaluation.")
    else:
        if is_regression:
            metric = evaluate.load("mse")
            logger.info("Using mean squared error (mse) as regression score, you can use --metric_name to overwrite.")
        else:
            if is_multi_label:
                f1_metric = evaluate.load("f1",config_name="multilabel")
                accuracy_metric = evaluate.load("accuracy",config_name="multilabel")
                precision_metric = evaluate.load("precision",config_name="multilabel")
                recall_metric = evaluate.load("recall",config_name="multilabel")
                logger.info(
                    "Using multilabel F1, accuracy, precision, recall for multi-label classification task, you can use --metric_name to overwrite."
                )
            else:
                f1_metric = evaluate.load("f1")
                accuracy_metric = evaluate.load("accuracy")
                precision_metric = evaluate.load("precision")
                recall_metric = evaluate.load("recall")
                logger.info("Using F1, accuracy, precision, recall as classification score, you can use --metric_name to overwrite.")
                
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if is_regression:
            preds = np.squeeze(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
        else:
            preds = np.argmax(preds, axis=1)
            accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
            micro_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="micro")["f1"]
            micro_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="micro")["precision"]
            micro_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="micro")["recall"]
            macro_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
            macro_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="macro")["precision"]
            macro_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="macro")["recall"]
            result = {
                "accuracy": accuracy,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "micro_precision": micro_precision,
                "macro_precision": macro_precision,
                "micro_recall": micro_recall,               
                "macro_recall": macro_recall
            }
        '''if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()'''
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        
        '''# Removing the `label` columns if exists because it might contains -1 and Trainer won't like that.
        if "label" in predict_dataset.features:
            predict_dataset = predict_dataset.remove_columns("label")'''
        #predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        metrics = trainer.predict(predict_dataset, metric_key_prefix="predict").metrics

        if is_regression:
            predictions = np.squeeze(predictions)
        elif is_multi_label:
            # Convert logits to multi-hot encoding. We compare the logits to 0 instead of 0.5, because the sigmoid is not applied.
            # You can also pass `preprocess_logits_for_metrics=lambda logits, labels: nn.functional.sigmoid(logits)` to the Trainer
            # and set p > 0.5 below (less efficient in this case)
            predictions = np.array([np.where(p > 0, 1, 0) for p in predictions])
        else:
            predictions = np.argmax(predictions, axis=1)


        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info("***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    elif is_multi_label:
                        # recover from multi-hot encoding
                        item = [label_list[i] for i in range(len(item)) if item[i] == 1]
                        writer.write(f"{index}\t{item}\n")
                    else:
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")
        logger.info("Predict results saved at {}".format(output_predict_file))



    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()
