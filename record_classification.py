import pandas as pd
import os
import json
import numpy as np
import scipy.stats as stats
#/home/weisi/TemporalAssessment/logs/AmzReviewHC_sentiment/bert-base-uncased/AY_T4/split_seed2_model_seed_42_3e-5/predict_results.json
Task= 'AmzReviewHC_sentiment' #'AmzReviewHC_sentiment'#'Mimic'
base_path = "/HDD16TB/weisi/logs"
#options = ['T1_T1', 'T1_T2', 'T1_T3', 'T1_T4', 'T2_T2', 'T2_T3', 'T2_T4', 'T3_T3', 'T3_T4', 'T4_T4', 'AY_T4']
options = ['T1-T1', 'T1-T2', 'T1-T3', 'T1-T4', 'T2-T1','T2-T2', 'T2-T3', 'T2-T4',  'T3-T1', 'T3-T2','T3-T3', 'T3-T4','T4-T1','T4-T2','T4-T3', 'T4-T4', 'AY-T4']
model_name = 'bert-base-uncased'
seeds = range(1, 6)

keys = [
    "predict_accuracy", "predict_macro_f1",
    "predict_macro_precision", "predict_macro_recall",
    "predict_micro_f1", "predict_micro_precision", "predict_micro_recall",
    "predict_weighted_f1", "predict_weighted_precision", "predict_weighted_recall", "predict_loss"
]


all_results = {option: {key: [] for key in keys} for option in options}

# calculate mean and 95%CI for each key
for option in options:
    for seed in seeds:
        file_path = os.path.join(base_path,Task, model_name,option.replace('-', '_'), f"split_seed{seed}", "model_seed_42_3e-5_20epc", "predict_results.json") #model_seed_42_3e-5_20epc
        with open(file_path, 'r') as file:
            data = json.load(file)
            for key in keys:
                all_results[option][key].append(data[key])

summary_results = {option: {} for option in options}

for option, results in all_results.items():
    for key, values in results.items():
        mean = np.mean(values)
        sem = stats.sem(values) # standard error
        ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=sem)  # 95%CI
        #summary_results[option][key] = f"{mean} ± {ci[1] - mean}"
        summary_results[option][key] = f"{mean:.2f} ± {ci[1] - mean:.2f}"


df_results = pd.DataFrame(summary_results).T

csv_file_path =os.path.join(base_path,Task,f"{model_name}_summary_results_seed1-5_new.csv")
df_results.to_csv(csv_file_path)

 

