export PYTHONPATH="/home/weisi/TemporalAssessment:$PYTHONPATH"
GPU_NUMBER="0,1"
MODEL_NAME='bert-base-uncased'  #bluebert 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12' ; 'roberta-base'
TASK='Mimic'
BATCH_SIZE=32
ACCUMULATION_STEPS=2
SPLITSEED=5
RUN_FILE='/home/weisi/TemporalAssessment/baselines/classification.py'
BASE_PATH='/home/weisi/TemporalAssessment/data/MIMIC-IV-Note'
#mimic-T1_2008_2010  mimic-T2_2011_2013  T3_2014_2016  T4_2017_2019  AY_2008_2016
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/AY_T4/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/mimic-AY_2008_2016-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/mimic-AY_2008_2016-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/mimic-T4_2017_2019-test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T1_T1/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/mimic-T1_2008_2010-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/mimic-T1_2008_2010-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/mimic-T1_2008_2010-test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1  --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T1_T2/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/mimic-T1_2008_2010-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/mimic-T1_2008_2010-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/mimic-T2_2011_2013-test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1  --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T1_T3/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/mimic-T1_2008_2010-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/mimic-T1_2008_2010-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/mimic-T3_2014_2016-test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1  --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T1_T4/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/mimic-T1_2008_2010-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/mimic-T1_2008_2010-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/mimic-T4_2017_2019-test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1  --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}



CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T2_T2/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/mimic-T2_2011_2013-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/mimic-T2_2011_2013-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/mimic-T2_2011_2013-test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1  --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T2_T3/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/mimic-T2_2011_2013-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/mimic-T2_2011_2013-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/mimic-T3_2014_2016-test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T2_T4/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/mimic-T2_2011_2013-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/mimic-T2_2011_2013-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/mimic-T4_2017_2019-test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T3_T3/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/mimic-T3_2014_2016-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/mimic-T3_2014_2016-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/mimic-T3_2014_2016-test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T3_T4/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/mimic-T3_2014_2016-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/mimic-T3_2014_2016-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/mimic-T4_2017_2019-test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T4_T4/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/mimic-T4_2017_2019-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/mimic-T4_2017_2019-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/mimic-T4_2017_2019-test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
