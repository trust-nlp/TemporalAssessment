GPU_NUMBER="0,1"
MODEL_NAME='bert-base-uncased'  #bluebert 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12' ; 'roberta-base'
TASK='mimic_notes'
BATCH_SIZE=32
ACCUMULATION_STEPS=2
SPLITSEED=2
RUN_FILE='/home/weisi/Temporal/baselines/classification.py'
BASE_PATH='/home/weisi/Temporal/data/MIMIC-IV-Note/seed2'



CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T1_T1/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/mimic_T1_2008-2010_train.json"\
    --validation_file  "$BASE_PATH/mimic_T1_2008-2010_validation.json"\
    --test_file  "$BASE_PATH/mimic_T1_2008-2010_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T1_T2/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/mimic_T1_2008-2010_train.json"\
    --validation_file  "$BASE_PATH/mimic_T1_2008-2010_validation.json"\
    --test_file  "$BASE_PATH/mimic_T2_2011-2013_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T1_T3/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/mimic_T1_2008-2010_train.json"\
    --validation_file  "$BASE_PATH/mimic_T1_2008-2010_validation.json"\
    --test_file  "$BASE_PATH/mimic_T3_2014-2016_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T1_T4/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/mimic_T1_2008-2010_train.json"\
    --validation_file  "$BASE_PATH/mimic_T1_2008-2010_validation.json"\
    --test_file  "$BASE_PATH/mimic_T4_2017-2019_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T2/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/mimic_T2_2011-2013_train.json"\
    --validation_file  "$BASE_PATH/mimic_T2_2011-2013_validation.json"\
    --test_file  "$BASE_PATH/mimic_T2_2011-2013_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T3/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/mimic_T2_2011-2013_train.json"\
    --validation_file  "$BASE_PATH/mimic_T2_2011-2013_validation.json"\
    --test_file  "$BASE_PATH/mimic_T3_2014-2016_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T4/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/mimic_T2_2011-2013_train.json"\
    --validation_file  "$BASE_PATH/mimic_T2_2011-2013_validation.json"\
    --test_file  "$BASE_PATH/mimic_T4_2017-2019_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T3_T3/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/mimic_T3_2014-2016_train.json"\
    --validation_file  "$BASE_PATH/mimic_T3_2014-2016_validation.json"\
    --test_file  "$BASE_PATH/mimic_T3_2014-2016_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T3_T4/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/mimic_T3_2014-2016_train.json"\
    --validation_file  "$BASE_PATH/mimic_T3_2014-2016_validation.json"\
    --test_file  "$BASE_PATH/mimic_T4_2017-2019_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T4_T4/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/mimic_T4_2017-2019_train.json"\
    --validation_file  "$BASE_PATH/mimic_T4_2017-2019_validation.json"\
    --test_file  "$BASE_PATH/mimic_T4_2017-2019_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

