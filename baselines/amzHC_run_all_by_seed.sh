GPU_NUMBER="0,1"
MODEL_NAME='bert-base-uncased'  #bluebert 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12' ; 'roberta-base'
TASK='amzReviewHC_rating'
BATCH_SIZE=32
ACCUMULATION_STEPS=2
SPLITSEED=1
RUN_FILE='/home/weisi/Temporal/baselines/classification.py'
BASE_PATH='/home/weisi/Temporal/data/Amazon/HealthCare'

#--label_column_name sentiment if want to use sentiment as label, or overall. defalt is label(1-5 rating)
#T1-T1 T1-T2 T1-T3 T2-T2 T2-T3 T3-T3 AY-T3

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T1_T1/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2010_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2010_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2010_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --label_column_name overall  --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T1_T2/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2010_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2010_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2011-2012_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --label_column_name overall --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T1_T3/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2010_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2010_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2013-2014_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --label_column_name overall --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T2/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2011-2012_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2011-2012_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2011-2012_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --label_column_name overall  --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T3/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2011-2012_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2011-2012_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2013-2014_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --label_column_name overall --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T3_T3/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2013-2014_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2013-2014_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2013-2014_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --label_column_name overall  --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/AY_T3/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_AY_2007-2014_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_AY_2007-2014_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2013-2014_test.json"\
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 25 \
    --shuffle_train_dataset --text_column_name text --label_column_name overall  --do_train --do_eval --do_predict --load_best_model_at_end --metric_for_best_model micro_f1 --greater_is_better True \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
