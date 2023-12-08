GPU_NUMBER="1"  #"0,1"
RUN_FILE='/home/weisi/TemporalAssessment/baselines/QA_seq2seq.py'
BASE_PATH='/home/weisi/TemporalAssessment/data/BIONLP/newsplit'
MODEL_NAME='t5-base' #'razent/SciFive-base-Pubmed_PMC' 't5-base'
BATCH_SIZE=32 #32?
ACCUMULATION_STEPS=2
SPLITSEED=2
TASK='BIONLP_newsplit'

#AY-T4 T1-T1 T1-T2 T1-T3 T1-T4 T2-T2 T2-T3 T2-T4 T3-T3 T3-T4 T4-T4 

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/AY_T4/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/bionlp_AY_2014-2018_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/bionlp_AY_2014-2018_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T4_2019_test.json"\
    --context_column context  --question_column question  --answer_column answer --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --num_train_epochs 25 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T1_T1/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/bionlp_T1_2014-2016_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T1_2014-2016_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T1_2014-2016_test.json"\
    --context_column context  --question_column question  --answer_column answer --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --num_train_epochs 25 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T1_T2/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/bionlp_T1_2014-2016_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T1_2014-2016_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T2_2017_test.json"\
    --context_column context  --question_column question  --answer_column answer --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --num_train_epochs 25 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T1_T3/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/bionlp_T1_2014-2016_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T1_2014-2016_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T3_2018_test.json"\
    --context_column context  --question_column question  --answer_column answer --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --num_train_epochs 25 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T1_T4/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/bionlp_T1_2014-2016_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T1_2014-2016_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T4_2019_test.json"\
    --context_column context  --question_column question  --answer_column answer --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --num_train_epochs 25 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T2/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/bionlp_T2_2017_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T2_2017_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T2_2017_test.json"\
    --context_column context  --question_column question  --answer_column answer --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --num_train_epochs 25 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T3/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/bionlp_T2_2017_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T2_2017_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T3_2018_test.json"\
    --context_column context  --question_column question  --answer_column answer --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --num_train_epochs 25 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T4/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/bionlp_T2_2017_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T2_2017_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T4_2019_test.json"\
    --context_column context  --question_column question  --answer_column answer --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --num_train_epochs 25 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T3_T3/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/bionlp_T3_2018_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T3_2018_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T3_2018_test.json"\
    --context_column context  --question_column question  --answer_column answer --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --num_train_epochs 25 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T3_T4/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/bionlp_T3_2018_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T3_2018_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T4_2019_test.json"\
    --context_column context  --question_column question  --answer_column answer --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --num_train_epochs 25 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T4_T4/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5\
    --train_file "$BASE_PATH/seed$SPLITSEED/bionlp_T4_2019_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T4_2019_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/bionlp_T4_2019_test.json"\
    --context_column context  --question_column question  --answer_column answer --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --num_train_epochs 25 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}




  