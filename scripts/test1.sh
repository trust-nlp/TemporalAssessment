GPU_NUMBER="0,1" #"0,1"
RUN_FILE='/home/weisi/TemporalAssessment/baselines/QA_seq2seq_BioASQ.py' #ideal ans of all question type can use _factoid.py the difference is in exact ans
BASE_PATH='/home/weisi/TemporalAssessment/data/BIOASQ_formatted'
MODEL_NAME='t5-base' #'razent/SciFive-base-Pubmed_PMC' 't5-base'
BATCH_SIZE=4 #32?
ACCUMULATION_STEPS=1
#SPLITSEED=1
QUES_TYPE='factoid'   #'yesno'
ANS_COL='exact_answer'
TASK='BioASQ_factoid_exact'

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir /home/weisi/TemporalAssessment/logs/${TASK}/${MODEL_NAME}/T1_T1/split_seed1/test\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T1_2013_2015-train.json"\
    --test_file  "$BASE_PATH/$QUES_TYPE/seed1/bioasq-$QUES_TYPE-T1_2013_2015-test.json"\
    --context_column snippets  --question_column body  --answer_column ${ANS_COL} --do_predict \
    --predict_with_generate --evaluation_strategy no --learning_rate 3e-4 --max_seq_length 384 --max_answer_length 20 --doc_stride 128 --num_train_epochs 0 \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
