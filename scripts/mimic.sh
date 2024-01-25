GPU_NUMBER="0,1"
TRAIN_PATH='/home/weisi/Temporal/data/MIMIC-IV-Note/mimic_Allyear_sampled_train.json'
VALIDATION_PATH='/home/weisi/Temporal/data/MIMIC-IV-Note/mimic_Allyear_sampled_test.json'
TEST_PATH='/home/weisi/Temporal/data/MIMIC-IV-Note/mimic_Allyear_sampled_validation.json'
MODEL_NAME='roberta-base' #bluebert 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12' ; 'bert-base-uncased' 
BATCH_SIZE=32
ACCUMULATION_STEPS=2
TASK='mimic_notes'
YEAR='Allyear_sampled'
# 4 time periods T1 T2 T3 T4；Tx&Ty use Tx_train for train and Tx_test for validation and test on Ty_train
#
#    --metric_name f1 \ will use only f1 for metrics

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python /home/weisi/Temporal/baselines/classification.py \
    --model_name_or_path ${MODEL_NAME} \
    --train_file ${TRAIN_PATH}\
    --validation_file  ${VALIDATION_PATH}\
    --test_file  ${TEST_PATH}\
    --seed 42  \
    --output_dir logs/${TASK}/${YEAR}/${MODEL_NAME}/seed_42_3e-5\
    --shuffle_train_dataset \
    --text_column_name text \
    --do_train \
    --do_eval \
    --do_predict \
    --load_best_model_at_end \
    --metric_for_best_model micro_f1 \
    --greater_is_better True \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --max_seq_length 512 \
    --learning_rate 3e-5 \
    --num_train_epochs 15 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${ACCUMULATION_STEPS}