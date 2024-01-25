GPU_NUMBER="0,1"
TRAIN_PATH='/home/weisi/Temporal/data/amazonQA/HC/QA_amazonHC_YN_2013_train.json'
TEST_PATH='/home/weisi/Temporal/data/amazonQA/HC/QA_amazonHC_YN_2015.json'
VALIDATION_PATH='/home/weisi/Temporal/data/amazonQA/HC/QA_amazonHC_YN_2013_validation.json'
MODEL_NAME='t5-small' 
BATCH_SIZE=32 #32?
ACCUMULATION_STEPS=2
TASK='amazonQA'
CATEGORY='Health_Personal_Care'



CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python /home/weisi/Temporal/baselines/QA_seq2seq.py \
  --model_name_or_path ${MODEL_NAME} \
  --context_column description \
  --question_column question \
  --answer_column answer \
  --train_file ${TRAIN_PATH}\
  --validation_file  ${VALIDATION_PATH}\
  --test_file  ${TEST_PATH}\
  --do_train \
  --do_eval \
  --do_predict \
  --predict_with_generate \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
  --learning_rate 3e-5 \
  --num_train_epochs 20 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --seed 42  \
  --output_dir logs/${TASK}/${CATEGORY}/${MODEL_NAME}/seed_42 