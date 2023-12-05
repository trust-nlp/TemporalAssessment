import json
import os
import pandas as pd

# paths
base_dir = '/home/weisi/TemporalAssessment/logs/bionlpQA'
options = ['T1-T1', 'T1-T2', 'T1-T3', 'T1-T4', 'T2-T2', 'T2-T3', 'T2-T4', 'T3-T3', 'T3-T4', 'T4-T4', 'AY-T4']
file_name = 'predict_results.json'


results = {}


for option in options:
    file_path = os.path.join(base_dir, option.replace('-', '_'), 't5-base', 'split_seed1_model_seed_42_3e-5', file_name)
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
df.to_csv('scores.csv')
