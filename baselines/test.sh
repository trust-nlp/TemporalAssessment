export PYTHONPATH="/home/weisi/TemporalAssessment:$PYTHONPATH"

GPU_NUMBER="0,1"
RUN_FILE='/home/weisi/TemporalAssessment/baselines/ner.py'
BASE_PATH='/home/weisi/TemporalAssessment/data/WIESP'
MODEL_NAME='bert-base-cased' # bert-base-uncased  roberta-base
BATCH_SIZE=16
ACCUMULATION_STEPS=2
TASK='WiespNER'

#AY-T4 T1-T1 T1-T2 T1-T3 T1-T4 T2-T2 T2-T3 T2-T4 T3-T3 T3-T4 T4-T4 
# T1_2013-2015 T2_2016-2018 T3_2019-2020 T4_2021-2022 AY_2013-2020


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T3/${MODEL_NAME}/split_seed1/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed1/wiesp-T2_2017_2018-train.json"\
    --validation_file  "$BASE_PATH/seed1/wiesp-T2_2017_2018-validation.json"\
    --test_file  "$BASE_PATH/seed1/wiesp-T3_2019-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T3/${MODEL_NAME}/split_seed2/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/wiesp-T2_2017_2018-train.json"\
    --validation_file  "$BASE_PATH/seed2/wiesp-T2_2017_2018-validation.json"\
    --test_file  "$BASE_PATH/seed2/wiesp-T3_2019-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T3/${MODEL_NAME}/split_seed3/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/wiesp-T2_2017_2018-train.json"\
    --validation_file  "$BASE_PATH/seed3/wiesp-T2_2017_2018-validation.json"\
    --test_file  "$BASE_PATH/seed3/wiesp-T3_2019-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T3/${MODEL_NAME}/split_seed4/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/wiesp-T2_2017_2018-train.json"\
    --validation_file  "$BASE_PATH/seed4/wiesp-T2_2017_2018-validation.json"\
    --test_file  "$BASE_PATH/seed4/wiesp-T3_2019-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T3/${MODEL_NAME}/split_seed5/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/wiesp-T2_2017_2018-train.json"\
    --validation_file  "$BASE_PATH/seed5/wiesp-T2_2017_2018-validation.json"\
    --test_file  "$BASE_PATH/seed5/wiesp-T3_2019-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
