export PYTHONPATH="/home/weisi/TemporalAssessment:$PYTHONPATH"

GPU_NUMBER="0,1"
RUN_FILE='/home/weisi/TemporalAssessment/baselines/ner.py'
BASE_PATH='/home/weisi/TemporalAssessment/data/GMB'
MODEL_NAME='bert-base-cased' # bert-base-uncased  roberta-base
BATCH_SIZE=32
ACCUMULATION_STEPS=2
SPLITSEED=1
TASK='GmbNER'
#sh /home/weisi/TemporalAssessment/scripts/gmb_run_all.sh
# 4 time periods 'gmb-T1_2004_2005 T2_2006_2007 T3_2008_2009 T4_2010_2011 AY_2004_2009

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/AY_T4/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/gmb-AY_2004_2009-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/gmb-AY_2004_2009-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/gmb-T4_2010_2011-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
    
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T1_T1/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/gmb-T1_2004_2005-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/gmb-T1_2004_2005-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/gmb-T1_2004_2005-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T1_T2/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/gmb-T1_2004_2005-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/gmb-T1_2004_2005-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/gmb-T2_2006_2007-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T1_T3/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/gmb-T1_2004_2005-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/gmb-T1_2004_2005-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/gmb-T3_2008_2009-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T1_T4/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/gmb-T1_2004_2005-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/gmb-T1_2004_2005-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/gmb-T4_2010_2011-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T2/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/gmb-T2_2006_2007-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/gmb-T2_2006_2007-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/gmb-T2_2006_2007-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T3/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/gmb-T2_2006_2007-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/gmb-T2_2006_2007-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/gmb-T3_2008_2009-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T2_T4/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/gmb-T2_2006_2007-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/gmb-T2_2006_2007-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/gmb-T4_2010_2011-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T3_T3/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/gmb-T3_2008_2009-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/gmb-T3_2008_2009-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/gmb-T3_2008_2009-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T3_T4/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/gmb-T3_2008_2009-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/gmb-T3_2008_2009-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/gmb-T4_2010_2011-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/T4_T4/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/gmb-T4_2010_2011-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/gmb-T4_2010_2011-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/gmb-T4_2010_2011-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


