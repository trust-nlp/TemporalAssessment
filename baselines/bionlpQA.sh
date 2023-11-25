GPU_NUMBER="1"
TRAIN_PATH='/home/weisi/Temporal/data/BIONLP/bionlp_2017_train.json'
TEST_PATH='/home/weisi/Temporal/data/BIONLP/bionlp_2017_test.json'
VALIDATION_PATH='/home/weisi/Temporal/data/BIONLP/bionlp_2017_val.json'
MODEL_NAME='razent/SciFive-base-Pubmed_PMC' #'t5-small' 
BATCH_SIZE=16 #32?
ACCUMULATION_STEPS=2
TASK='bionlpQA'
YEAR='2017' #train17test18: train in 20xx test in 20xx; 20xx: train and test in 20xx




CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python /home/weisi/Temporal/baselines/QA_seq2seq.py \
  --model_name_or_path ${MODEL_NAME} \
  --context_column context \
  --question_column question \
  --answer_column answer \
  --train_file ${TRAIN_PATH}\
  --validation_file  ${VALIDATION_PATH}\
  --test_file  ${TEST_PATH}\
  --do_train \
  --do_eval \
  --do_predict\
  --predict_with_generate \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --seed 13  \
  --output_dir logs/${TASK}/${YEAR}/${MODEL_NAME}/seed_13