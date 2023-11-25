GPU_NUMBER="0,1"
TRAIN_PATH='/home/weisi/Temporal/data/WIESP/WIESP2022-NER-train.json'
VALIDATION_PATH='/home/weisi/Temporal/data/WIESP/WIESP2022-NER-validation.json'
TEST_PATH='/home/weisi/Temporal/data/WIESP/WIESP2022-NER-test.json'
MODEL_NAME='bert-base-cased' # roberta-base
BATCH_SIZE=32
ACCUMULATION_STEPS=2
TASK='wiespNER'
YEAR='allyears'
# 3 time periods T1 T2 T3；Tx&Ty use Tx_train for train and Tx_test for validation and test on Ty_train


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python /home/weisi/Temporal/baselines/wiesp.py \
  --model_name_or_path ${MODEL_NAME} \
  --train_file ${TRAIN_PATH}\
  --validation_file  ${VALIDATION_PATH}\
  --test_file  ${TEST_PATH}\
  --seed 42  \
  --output_dir logs/${TASK}/${YEAR}/${MODEL_NAME}/seed_42 \
  --do_train --do_eval --do_pred \
  --overwrite_output_dir \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --greater_is_better True \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --num_train_epochs 20 \
  --learning_rate 3e-5 \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
  --eval_accumulation_steps ${ACCUMULATION_STEPS}
