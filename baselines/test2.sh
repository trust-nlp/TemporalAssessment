export PYTHONPATH="/home/weisi/TemporalAssessment:$PYTHONPATH"
GPU_NUMBER=1 #"0,1"
MODEL_NAME='bert-base-uncased'
#'xdai/mimic_roberta_base'  1
# rjac/biobert-ICD10-L3-mimic 1 'bert-base-uncased'1,
# bluebert 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'1 ;
#'medicalai/ClinicalBERT'1
#emilyalsentzer/Bio_ClinicalBERT 1
TASK='Mimic' 
BATCH_SIZE=32
ACCUMULATION_STEPS=2
SPLITSEED=1
RUN_FILE='/home/weisi/TemporalAssessment/baselines/classification.py'
BASE_PATH='/home/weisi/TemporalAssessment/data/MIMIC-IV-Note'


CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ${RUN_FILE} --model_name_or_path ${MODEL_NAME} --seed 42\
    --output_dir HDD16TB/weisi/logs/${TASK}/${MODEL_NAME}/T4_T4/split_seed${SPLITSEED}model_seed_42_3e-5_20epc\
    --overwrite_output_dir \
    --train_file "$BASE_PATH/seed$SPLITSEED/mimic-T4_2017_2019-train.json"\
    --validation_file  "$BASE_PATH/seed$SPLITSEED/mimic-T4_2017_2019-validation.json"\
    --test_file  "$BASE_PATH/seed$SPLITSEED/mimic-T4_2017_2019-test.json"\
    --evaluation_strategy epoch --save_strategy no --max_seq_length 512 --learning_rate 3e-5 --num_train_epochs 20 \
    --shuffle_train_dataset --text_column_names text --label_column_name label  --do_train --do_eval --do_predict \
    --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}  
