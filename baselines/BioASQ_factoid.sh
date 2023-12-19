GPU_NUMBER="1"  #"0,1"
RUN_FILE='/home/weisi/TemporalAssessment/baselines/QA_seq2seq_factoid.py'
BASE_PATH='/home/weisi/TemporalAssessment/data/BIOASQ'
MODEL_NAME='razent/SciFive-base-Pubmed_PMC' #'razent/SciFive-base-Pubmed_PMC' 't5-base'
BATCH_SIZE=32 #32?
ACCUMULATION_STEPS=2
SPLITSEED=1
TASK='BioASQ_factoid_exact'
#bioasq_factoid_T1_2013-2015_train.json
#AY-T4 T1-T1 T1-T2 T1-T3 T1-T4 T2-T2 T2-T3 T2-T4 T3-T3 T3-T4 T4-T4 

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir logs/${TASK}/AY_T4/${MODEL_NAME}/split_seed${SPLITSEED}_model_seed_42_3e-5_maxanslen20\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/bioasq_factoid_AY_2013-2020_train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/bioasq_factoid_AY_2013-2020_validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/bioasq_factoid_T4_2021-2022_test.json"\
    --context_column snippets  --question_column body  --answer_column exact_answer --do_train --do_eval --do_predict \
    --predict_with_generate --evaluation_strategy epoch --learning_rate 3e-5 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 25 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
