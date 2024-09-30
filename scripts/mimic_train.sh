export PYTHONPATH="/home/weisi/TemporalAssessment:$PYTHONPATH"
GPU_NUMBER=1 #"0,1"
MODEL_NAME='roberta-base'  #allenai/longformer-base-4096 bluebert 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12' ; 'roberta-base'
TASK='Mimic_icd10'
BATCH_SIZE=32
ACCUMULATION_STEPS=2
SPLITSEED=1
MODELSEED=42
RUN_FILE='/home/weisi/TemporalAssessment/baselines/classification.py'
BASE_PATH='/home/weisi/TemporalAssessment/data/MIMIC-IV-Note'


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed ${MODELSEED}\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T1/split_seed${SPLITSEED}/model_seed_${MODELSEED}_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/T1_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/T1_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/T1_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed ${MODELSEED}\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T2/split_seed${SPLITSEED}/model_seed_${MODELSEED}_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/T2_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/T2_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/T2_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed ${MODELSEED}\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T3/split_seed${SPLITSEED}/model_seed_${MODELSEED}_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/T3_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/T3_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/T3_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed ${MODELSEED}\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T4/split_seed${SPLITSEED}/model_seed_${MODELSEED}_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/T4_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/T4_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/T4_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
