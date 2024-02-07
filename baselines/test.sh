export PYTHONPATH="/home/weisi/TemporalAssessment:$PYTHONPATH"
GPU_NUMBER="0,1"
MODEL_NAME='bert-base-uncased'  #bluebert 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12' ; 'roberta-base'
TASK='Mimic'
BATCH_SIZE=32
ACCUMULATION_STEPS=2
SPLITSEED=1

RUN_FILE='/home/weisi/TemporalAssessment/baselines/classification.py'
BASE_PATH='/home/weisi/TemporalAssessment/data/MIMIC-IV-Note'

#AY-T4 T1-T1 T1-T2 T1-T3 T1-T4 T2-T2 T2-T3 T2-T4 T3-T3 T3-T4 T4-T4 
# T1_2013-2015 T2_2016-2018 T3_2019-2020 T4_2021-2022 AY_2013-2020
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T4_T4/split_seed1/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed1/mimic-T4_2017_2019-train.json"\
    --validation_file  "$BASE_PATH/seed1/mimic-T4_2017_2019-validation.json"\
    --test_file  "$BASE_PATH/seed1/mimic-T4_2017_2019-test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T4_T4/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/mimic-T4_2017_2019-train.json"\
    --validation_file  "$BASE_PATH/seed2/mimic-T4_2017_2019-validation.json"\
    --test_file  "$BASE_PATH/seed2/mimic-T4_2017_2019-test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T4_T4/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/mimic-T4_2017_2019-train.json"\
    --validation_file  "$BASE_PATH/seed3/mimic-T4_2017_2019-validation.json"\
    --test_file  "$BASE_PATH/seed3/mimic-T4_2017_2019-test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/AY_T4/split_seed1/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed1/mimic-AY_2008_2016-train.json"\
    --validation_file  "$BASE_PATH/seed1/mimic-AY_2008_2016-validation.json"\
    --test_file  "$BASE_PATH/seed1/mimic-T4_2017_2019-test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
