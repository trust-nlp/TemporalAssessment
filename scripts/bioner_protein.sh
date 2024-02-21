export PYTHONPATH="/home/weisi/TemporalAssessment:$PYTHONPATH"

GPU_NUMBER="0,1"
RUN_FILE='/home/weisi/TemporalAssessment/baselines/ner.py'
BASE_PATH='/home/weisi/TemporalAssessment/data/BioNER/Protein'
MODEL_NAME='bert-base-cased' # bert-base-uncased  roberta-base
BATCH_SIZE=16
ACCUMULATION_STEPS=2
SPLITSEED=1
TASK='BioNER_Protein_IOBES'
#sh /home/weisi/TemporalAssessment/scripts/gmb_run_all.sh
# 4 'D1_09_iobes', 'D2_11EPI_iobes', 'D3_11ID_iobes', 'D4_13GE_iobes'


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D1/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D2/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D3/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D4/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D1/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
    
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D2/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D3/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D4/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D1/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D2/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D3/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D4/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D1/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D2/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D3/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D4/${MODEL_NAME}/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

## ------------ 2 ----------

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D1/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D2/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D3/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D4/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D1/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
    
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D2/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D3/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D4/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D1/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D2/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D3/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D4/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D1/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D2/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D3/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D4/${MODEL_NAME}/split_seed2/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed2/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed2/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed2/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

## ------------ 3----------


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D1/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D2/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D3/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D4/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D1/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
    
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D2/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D3/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D4/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D1/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D2/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D3/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D4/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D1/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D2/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D3/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D4/${MODEL_NAME}/split_seed3/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed3/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed3/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed3/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

## ------------ 4 ----------

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D1/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D2/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D3/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D4/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D1/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
    
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D2/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D3/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D4/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D1/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D2/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D3/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D4/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D1/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D2/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D3/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D4/${MODEL_NAME}/split_seed4/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed4/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed4/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed4/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

## ------------5----------


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D1/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D2/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D3/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D1_D4/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D1_09_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D1_09_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D1/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
    
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D2/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D3/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D2_D4/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D2_11EPI_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D2_11EPI_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D1/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D2/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D3/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D3_D4/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D3_11ID_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D3_11ID_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D1/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D1_09_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D2/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D2_11EPI_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D3/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D3_11ID_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/D4_D4/${MODEL_NAME}/split_seed5/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed5/D4_13GE_iobes-train.json"\
    --validation_file  "$BASE_PATH/seed5/D4_13GE_iobes-validation.json"\
    --test_file  "$BASE_PATH/seed5/D4_13GE_iobes-test.json"\
    --task_name ner --do_train --do_eval --do_predict \
    --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}






