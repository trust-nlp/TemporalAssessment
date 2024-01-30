import pandas as pd
import os
import json
import numpy as np
import scipy.stats as stats

Task='WiespNER'
base_path = "/home/weisi/TemporalAssessment/logs"
options = ['T1_T1', 'T1_T2', 'T1_T3', 'T1_T4', 'T2_T2', 'T2_T3', 'T2_T4', 'T3_T3', 'T3_T4', 'T4_T4', 'AY_T4']
model_name = 'bert-base-cased'
seeds = range(1, 6)

keys = [
    "predict_accuracy", "predict_loss", "predict_macro_f1",
    "predict_macro_precision", "predict_macro_recall",
    "predict_micro_f1", "predict_micro_precision", "predict_micro_recall",
    "predict_weighted_f1", "predict_weighted_precision", "predict_weighted_recall"
]


all_results = {option: {key: [] for key in keys} for option in options}

# calculate mean and 95%CI for each key
for option in options:
    for seed in seeds:
        file_path = os.path.join(base_path,Task, option, model_name, f"split_seed{seed}", "model_seed_42_3e-5", "predict_results.json")
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
        summary_results[option][key] = f"{mean} ± {ci[1] - mean}"


df_results = pd.DataFrame(summary_results).T

csv_file_path =os.path.join(base_path,Task,f"{model_name}_summary_results.csv")
df_results.to_csv(csv_file_path)

 

