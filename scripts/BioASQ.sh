GPU_NUMBER="0,1" #"0,1"
RUN_FILE='/home/weisi/TemporalAssessment/baselines/QA_seq2seq_BioASQ.py' #ideal ans of all question type can use _factoid.py the difference is in exact ans
BASE_PATH='/home/weisi/TemporalAssessment/data/BIOASQ'
MODEL_NAME='t5-base' #'razent/SciFive-base-Pubmed_PMC' 't5-base'
BATCH_SIZE=4 #32?
ACCUMULATION_STEPS=1
SPLITSEED=5
QUES_TYPE='factoid'   #'yesno'
ANS_COL='exact_answer'
TASK='BioASQ_factoid_exact'

#AY-T4 T1-T1 T1-T2 T1-T3 T1-T4 T2-T2 T2-T3 T2-T4 T3-T3 T3-T4 T4-T4 
# T1_2013-2015 T2_2016_2018 T3_2019_2020 T4_2021_2022 AY_2013_2020
#b4a1:batchsize4 accumulation step1
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/AY_T4/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-AY_2013_2020-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-AY_2013_2020-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T1_T1/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T1_2013_2015-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T1_T2/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T1_2013_2015-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T1_T3/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T1_2013_2015-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T1_T4/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T1_2013_2015-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T2_T2/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T2_2016_2018-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T2_T3/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T2_2016_2018-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T2_T4/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T2_2016_2018-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T3_T3/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T3_2019_2020-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T3_T4/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T3_2019_2020-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T4_T4/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T4_2021_2022-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

######part2 

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T4_T3/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T4_2021_2022-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T4_T2/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T4_2021_2022-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T4_T1/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T4_2021_2022-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T3_T2/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T3_2019_2020-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T3_T1/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T3_2019_2020-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T2_T1/split_seed${SPLITSEED}/b4a1_sd42_3e-4_maxanslen20_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --validation_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T2_2016_2018-validation.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed$SPLITSEED/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
