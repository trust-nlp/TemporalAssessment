import pandas as pd
import os
import json
import numpy as np
import scipy.stats as stats
import math

# paths
base_dir ='/HDD16TB/weisi/logs/BioASQ_alltypes_exact_new' #'/HDD16TB/weisi/logs/BioASQ_factoid_exact'#'/HDD16TB/weisi/logs/BioASQ_alltypes_exact'1#
out_csv_file_name = 'BioASQ_alltypes_exact-t5-base_seed1-5_new.csv'
options = ['T1-T1', 'T1-T2', 'T1-T3', 'T1-T4', 'T2-T1','T2-T2', 'T2-T3', 'T2-T4', 'T3-T1', 'T3-T2','T3-T3', 'T3-T4','T4-T1','T4-T2','T4-T3', 'T4-T4', 'ALL-ALL']
#options = ['T1-T1', 'T1-T2', 'T1-T3', 'T1-T4', 'T2-T1','T2-T2', 'T2-T3', 'T2-T4', 'T3-T1', 'T3-T2','T3-T3', 'T3-T4','T4-T1','T4-T2','T4-T3', 'T4-T4', 'AY-T4']
file_name = 'predict_results.json'
seeds = range(1, 6) 
results = {}

#razent/SciFive-base-Pubmed_PMC
#'t5-base'

# Metrics to summarize

#metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum','exp_of_average_log_n_precisions','bleu', 'brevity_penalty','1 precision', '2 precision', '3 precision', '4 precision', 'exact_match', 'google_bleu']
#metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'exact_match']

metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum','meteor','exp_of_average_log_n_precisions', 'exact_match','bleu', 'brevity_penalty','1 precision', '2 precision', '3 precision', '4 precision', 'google_bleu']

# Initialize dictionary to hold results
all_results = {option: {metric: [] for metric in metrics} for option in options}

# Process results for each option and seed
for option in options:
    for seed in seeds:
        file_path = os.path.join(base_dir, 't5-base', option.replace('-', '_'), f'split_seed{seed}', 'b4a1_sd42_3e-4_maxanslen30_20epc', 'predict_results.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
                all_results[option]['rouge1'].append(data['test_rouge']['rouge1'])
                all_results[option]['rouge2'].append(data['test_rouge']['rouge2'])
                all_results[option]['rougeL'].append(data['test_rouge']['rougeL'])
                all_results[option]['rougeLsum'].append(data['test_rouge']['rougeLsum'])
                all_results[option]['meteor'].append(data['test_meteor']['meteor'])
                all_results[option]['exact_match'].append(data['test_exact_match']['exact_match'])
                precisions = data['test_bleu']['precisions']  
                log_precisions = [math.log(p) for p in precisions if p > 0] 
                avg_log_precision = sum(log_precisions) / len(log_precisions)
                exp_of_avg_log_precision = math.exp(avg_log_precision)
                all_results[option]['exp_of_average_log_n_precisions'].append(exp_of_avg_log_precision)

                
                all_results[option]['bleu'].append(data['test_bleu']['bleu'])
                all_results[option]['brevity_penalty'].append(data['test_bleu']['brevity_penalty'])
                all_results[option]['1 precision'].append(data['test_bleu']['precisions'][0])  # 1-gram precision
                all_results[option]['2 precision'].append(data['test_bleu']['precisions'][1])  # 2-gram precision
                all_results[option]['3 precision'].append(data['test_bleu']['precisions'][2])  # 3-gram precision
                all_results[option]['4 precision'].append(data['test_bleu']['precisions'][3])  # 4-gram precision
                all_results[option]['google_bleu'].append(data['test_google_bleu']['google_bleu'])
                

# Calculate summary statistics
summary_results = {option: {} for option in options}
#print(all_results)
for option, results in all_results.items():
    for metric, values in results.items():
        print(option,metric, values)
        if values:  # Check if there are results to avoid division by zero
            mean = np.mean(values)
            sem = stats.sem(values)  # Standard error of the mean
            ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=sem)  # 95% CI
            summary_results[option][metric] = f"{mean:.2f} ± {ci[1] - mean:.2f}"

# Convert summary to DataFrame
df_summary = pd.DataFrame(summary_results).T

# Save summary to CSV
df_summary.to_csv(os.path.join(base_dir, out_csv_file_name))

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
