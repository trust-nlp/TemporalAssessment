GPU_NUMBER="0,1" #"0,1"
RUN_FILE='/home/weisi/TemporalAssessment/baselines/QA_seq2seq_BioASQ.py' 
BASE_PATH='/home/weisi/TemporalAssessment/data/BIOASQ_formatted'
MODEL_NAME='t5-large' #'razent/SciFive-base-Pubmed_PMC' 't5-base'
BATCH_SIZE=4 #32?
ACCUMULATION_STEPS=1
#SPLITSEED=1 
QUES_TYPE='alltypes'  #'yesno'1 'list'new1 record 'factoid'new1 'summary'new1 alltypes
ANS_COL='exact_answer'
TASK='BioASQ_alltypes_exact'

#AY-T4 T1-T1 T1-T2 T1-T3 T1-T4 T2-T2 T2-T3 T2-T4 T3-T3 T3-T4 T4-T4 
# T1_2013_2015 T2_2016_2018 T3_2019_2020 T4_2021_2022 AY_2013_2020
#b4a1:batchsize4 accumulation 
#ALL: T1-T4
#AY: T1-T3
#in previous experiments, exact ans are not changed to right format for list summary and yesno question 
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/ALL_ALL/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-ALL_2013_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-ALL_2013_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T1/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T2/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T3/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T4/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T1/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T2/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T3/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T4/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T1/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T2/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T3/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T4/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T1/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T2/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T3/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T4/split_seed1/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

###### seed2 #########
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/ALL_ALL/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-ALL_2013_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-ALL_2013_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T1/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T2/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T3/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T4/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T1/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T2/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T3/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T4/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T1/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T2/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T3/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T4/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T1/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T2/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T3/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T4/split_seed2/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed2/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

######### seed3###########

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/ALL_ALL/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-ALL_2013_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-ALL_2013_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T1/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T2/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T3/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T4/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T1/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T2/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T3/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T4/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T1/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T2/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T3/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T4/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T1/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T2/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T3/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T4/split_seed3/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed3/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

######### seed4 #########
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/ALL_ALL/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-ALL_2013_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-ALL_2013_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T1/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T2/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T3/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T4/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T1/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T2/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T3/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T4/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T1/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T2/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T3/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T4/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T1/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T2/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T3/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T4/split_seed4/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed4/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

################seed5########################
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/ALL_ALL/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-ALL_2013_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-ALL_2013_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T1/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T2/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T3/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T1_T4/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T1/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T2/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T3/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T2_T4/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T2_2016_2018-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T1/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T2/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T3/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T3_T4/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T3_2019_2020-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T1/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T2/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T2_2016_2018-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T3/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T3_2019_2020-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T4/split_seed5/b4a1_sd42_3e-4_maxanslen30_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T4_2021_2022-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed5/bioasq-$QUES_TYPE-T4_2021_2022-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_train --do_predict \
    --predict_with_generate --evaluation_strategy no --save_strategy no  --learning_rate 3e-4 --max_seq_length 512 --max_answer_length 30 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
