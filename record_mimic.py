import pandas as pd
import os
import json
import numpy as np

Task='Mimic_icd10'

base_path = "/HDD16TB/weisi/logs"
options = [
    'T1-T1', 'T1-T2', 'T1-T3', 'T1-T4', 'T2-T1', 'T2-T2', 'T2-T3', 'T2-T4',
    'T3-T1', 'T3-T2', 'T3-T3', 'T3-T4', 'T4-T1', 'T4-T2', 'T4-T3', 'T4-T4']# 
model_name = 'roberta-base'
#model_seeds = ['41', '42', '43']
model_seeds = ['42']
keys = [
    "predict_accuracy", "predict_macro_f1",
    "predict_macro_precision", "predict_macro_recall",
    "predict_micro_f1", "predict_micro_precision", "predict_micro_recall",
    "predict_sample_f1", "predict_sample_precision", "predict_sample_recall",
    "predict_weighted_f1", "predict_weighted_precision", "predict_weighted_recall"
]

all_results = {option: {key: [] for key in keys} for option in options}

# calculate mean for each key
for option in options:
    for model_seed in model_seeds:
        file_path = os.path.join(base_path, Task, model_name, option.replace('-', '_'), "split_seed1", f"model_seed_{model_seed}_3e-5_20epc", "predict_results.json")
        with open(file_path, 'r') as file:
            data = json.load(file)
            for key in keys:
                all_results[option][key].append(data[key])

summary_results = {option: {} for option in options}

for option, results in all_results.items():
    for key, values in results.items():
        #mean = np.mean(values)
        mean = np.mean(values) * 100  # Convert to percentage
        summary_results[option][key] = f"{mean:.3f}"

df_results = pd.DataFrame(summary_results).T

csv_file_path = os.path.join(base_path, Task, f"{model_name}_summary_results_model_seeds_42.csv")
df_results.to_csv(csv_file_path)

