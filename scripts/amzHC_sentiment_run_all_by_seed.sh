export PYTHONPATH="/home/weisi/TemporalAssessment:$PYTHONPATH"
GPU_NUMBER="0,1"
MODEL_NAME='bert-base-uncased'  #bluebert 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12' ; 'roberta-base'
TASK='AmzReviewHC_sentiment'
BATCH_SIZE=32
ACCUMULATION_STEPS=2
SPLITSEED=4
RUN_FILE='/home/weisi/TemporalAssessment/baselines/classification.py'
BASE_PATH='/home/weisi/TemporalAssessment/data/Amazon/HealthCare'

#--label_column_name sentiment if want to use sentiment as label, or overall. defalt is label(1-5 rating)
#T1-T1 T1-T2 T1-T3 T1-T4 T2-T2 T2-T3 T2-T4 T3-T3 T3-T4 T4-T4 AY-T4

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T1_T1/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2008_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2008_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2008_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment  --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T1_T2/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2008_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2008_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2009-2010_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T1_T3/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2008_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2008_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2011-2012_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T1_T4/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2008_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2008_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T4_2013-2014_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T2_T2/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2009-2010_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2009-2010_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2009-2010_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment  --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T2_T3/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2009-2010_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2009-2010_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2011-2012_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T2_T4/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2009-2010_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2009-2010_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T4_2013-2014_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T3_T3/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2011-2012_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2011-2012_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2011-2012_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment  --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T3_T4/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2011-2012_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2011-2012_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T4_2013-2014_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment  --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T4_T4/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T4_2013-2014_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T4_2013-2014_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T4_2013-2014_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment  --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/AY_T4/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_AY_2007-2012_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_AY_2007-2012_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T4_2013-2014_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20\
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment  --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

##### part2

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T4_T3/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T4_2013-2014_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T4_2013-2014_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2011-2012_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment  --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T4_T2/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T4_2013-2014_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T4_2013-2014_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2009-2010_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment  --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T4_T1/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T4_2013-2014_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T4_2013-2014_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2008_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment  --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T3_T2/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2011-2012_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2011-2012_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2009-2010_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment  --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T3_T1/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2011-2012_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T3_2011-2012_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2008_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment  --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/${MODEL_NAME}/T2_T1/split_seed${SPLITSEED}/model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2009-2010_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T2_2009-2010_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/amzHC_T1_2007-2008_test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name sentiment --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

