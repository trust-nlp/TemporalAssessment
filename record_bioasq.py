import pandas as pd
import os
import json
import numpy as np
import scipy.stats as stats

# paths
base_dir = '/home/weisi/TemporalAssessment/logs/BioASQ_factoid_exact'
options = ['T1-T1', 'T1-T2', 'T1-T3', 'T1-T4', 'T2-T1','T2-T2', 'T2-T3', 'T2-T4',  'T3-T1', 'T3-T2','T3-T3', 'T3-T4','T4-T1','T4-T2','T4-T3', 'T4-T4', 'AY-T4']
file_name = 'predict_results.json'
out_csv_file_name = 'BioASQ_factoid_exact_t5-base_seed1-5.csv'
seeds = range(1, 6) 
results = {}

#razent/SciFive-base-Pubmed_PMC
#'t5-base'
#/home/weisi/TemporalAssessment/logs/BioASQ_factoid_exact/T1_T1/t5-base/split_seed1/model_seed_42_3e-5_maxanslen20
#/home/weisi/TemporalAssessment/logs/BioASQ_factoid_ideal/T1_T1/t5-base/split_seed1/b4a1_sd42_3e-4_maxanslen20
'''for option in options:
    file_path = os.path.join(base_dir,  't5-base', option.replace('-', '_'),'split_seed1','b4a1_sd42_3e-4_maxanslen20', file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            bleu_score = data['test_bleu']['bleu']
            precisions = data['test_bleu']['precisions'] # 4 n gram precision
            exact_match = data['test_exact_match']['exact_match']
            google_bleu = data['test_google_bleu']['google_bleu']
            rouge_scores = data['test_rouge']
            # add to result
            results[option] = [bleu_score] + precisions + [exact_match, google_bleu] + list(rouge_scores.values())
df = pd.DataFrame.from_dict(results, orient='index', columns=['bleu', '1 precision', '2 precision', '3 precision', '4 precision', 'exact_match', 'google_bleu', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
df.to_csv(os.path.join(base_dir,out_csv_file_name))
            '''


# Metrics to summarize
metrics = ['bleu', '1 precision', '2 precision', '3 precision', '4 precision', 'exact_match', 'google_bleu', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum']

# Initialize dictionary to hold results
all_results = {option: {metric: [] for metric in metrics} for option in options}

# Process results for each option and seed
for option in options:
    for seed in seeds:
        file_path = os.path.join(base_dir, 't5-base', option.replace('-', '_'), f'split_seed{seed}', 'b4a1_sd42_3e-4_maxanslen20_20epc', 'predict_results.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
                all_results[option]['bleu'].append(data['test_bleu']['bleu'])
                all_results[option]['1 precision'].append(data['test_bleu']['precisions'][0])  # 1-gram precision
                all_results[option]['2 precision'].append(data['test_bleu']['precisions'][1])  # 2-gram precision
                all_results[option]['3 precision'].append(data['test_bleu']['precisions'][2])  # 3-gram precision
                all_results[option]['4 precision'].append(data['test_bleu']['precisions'][3])  # 4-gram precision
                all_results[option]['exact_match'].append(data['test_exact_match']['exact_match'])
                all_results[option]['google_bleu'].append(data['test_google_bleu']['google_bleu'])
                all_results[option]['rouge1'].append(data['test_rouge']['rouge1'])
                all_results[option]['rouge2'].append(data['test_rouge']['rouge2'])
                all_results[option]['rougeL'].append(data['test_rouge']['rougeL'])
                all_results[option]['rougeLsum'].append(data['test_rouge']['rougeLsum'])

# Calculate summary statistics
summary_results = {option: {} for option in options}
for option, results in all_results.items():
    for metric, values in results.items():
        if values:  # Check if there are results to avoid division by zero
            mean = np.mean(values)
            sem = stats.sem(values)  # Standard error of the mean
            ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=sem)  # 95% CI
            summary_results[option][metric] = f"{mean:.2f} ± {ci[1] - mean:.2f}"

# Convert summary to DataFrame
df_summary = pd.DataFrame(summary_results).T

# Save summary to CSV
df_summary.to_csv(os.path.join(base_dir, out_csv_file_name))
