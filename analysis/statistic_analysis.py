import os
import json
import numpy as np

def load_performance_matrices(base_path, task, model_name, seeds, folder_name, metrics, file_name="predict_results.json"):
    performance_matrices = {metric: np.zeros((4, 4)) for metric in metrics}
    for metric in metrics:
        for i in range(1, 5):
            for j in range(1, 5):
                values = []
                for seed in seeds:
                    option = f'T{i}-T{j}'
                    file_path = os.path.join(base_path, task, model_name, option.replace('-', '_'), f"split_seed{seed}", folder_name, file_name)
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        values.append(data[metric])
                performance_matrices[metric][j-1, i-1] = np.mean(values)
    return performance_matrices

def compute_performance_change_matrices(performance_matrices):
    change_matrices = {}
    for metric, matrix in performance_matrices.items():
        change_matrix = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                change_matrix[i, j] = matrix[i, j] - matrix[i, i]  
        change_matrices[metric] = change_matrix
    return change_matrices

def load_token_or_embedding_matrix(path, keys):
    with open(path, 'r') as file:
        data = json.load(file)
    matrix_size = 4
    matrix = np.full((matrix_size, matrix_size), np.nan)
    for key in keys:
        i, j = map(int, key.split('_')[1:])
        matrix[i-1, j-1] = data[key]
        matrix[j-1, i-1] = data[key]
    return matrix.flatten()

def compute_correlations(performance_vecs, metric_vecs):
    correlation_matrix = {}
    for metric, perf_vec in performance_vecs.items():
        correlation_matrix[metric] = {key: np.corrcoef(perf_vec, vec)[0, 1] for key, vec in metric_vecs.items()}
    return correlation_matrix

# 参数和路径
base_path = "/HDD16TB/weisi/logs"
tasks = ['BioASQ_alltypes_exact_new', 'Mimic', 'BioNER_Protein_IOBES']
model_names = ['t5-base', 'bert-base-uncased', 'bert-base-cased']
folder_names = ['b4a1_sd42_3e-4_maxanslen30_20epc', 'model_seed_42_3e-5_20epc', 'model_seed_42_3e-5_20epc']
metrics = [["rougeL", "meteor", "Geometric_Average_Precision"], ["predict_micro_f1", "predict_micro_precision", "predict_micro_recall"]]
keys = ['T1_T1', 'T1_T2', 'T1_T3', 'T1_T4', 'T2_T2', 'T2_T3', 'T2_T4', 'T3_T3', 'T3_T4', 'T4_T4']
seeds = range(1, 6)

# 计算每个任务的performance matrices和change vectors
performance_data = {}
for task, model_name, folder_name, task_metrics in zip(tasks, model_names, folder_names, metrics):
    perf_matrices = load_performance_matrices(base_path, task, model_name, seeds, folder_name, task_metrics)
    perf_change_matrices = compute_performance_change_matrices(perf_matrices)
    performance_data[task] = {metric: matrix.flatten() for metric, matrix in perf_change_matrices.items()}

# 加载token和embedding相关矩阵
metric_paths = {
    'bioasq': {
        'jaccard': '/home/weisi/TemporalAssessment/analysis/token_metrics/bioasq_jaccard_results.json',
        'tfidf': '/home/weisi/TemporalAssessment/analysis/token_metrics/bioasq_cosine_tfidf_of_most_freq_tokens.json',
        'cosine_dist': '/home/weisi/TemporalAssessment/analysis/embedding_metrics_and_t-test/bioasq_miniLM_t-test-results.json',
        'euclidean_dist': '/home/weisi/TemporalAssessment/analysis/embedding_metrics_and_t-test/bioasq_use_t-test-results.json'
    },
    'mimic': {
        'jaccard': '/home/weisi/TemporalAssessment/analysis/token_metrics/mimic_jaccard_results.json',
        'tfidf': '/home/weisi/TemporalAssessment/analysis/token_metrics/mimic_cosine_tfidf_of_most_freq_tokens.json',
        'cosine_dist': '/home/weisi/TemporalAssessment/analysis/embedding_metrics_and_t-test/mimic_miniLM_t-test-results.json',
        'euclidean_dist': '/home/weisi/TemporalAssessment/analysis/embedding_metrics_and_t-test/mimic_use_t-test-results.json'
    },
    'bioner': {
        'jaccard': '/home/weisi/TemporalAssessment/analysis/token_metrics/bioner_jaccard_results.json',
        'tfidf': '/home/weisi/TemporalAssessment/analysis/token_metrics/bioner_cosine_tfidf_of_most_freq_tokens.json',
        'cosine_dist': '/home/weisi/TemporalAssessment/analysis/embedding_metrics_and_t-test/bioner_miniLM_t-test-results.json',
        'euclidean_dist': '/home/weisi/TemporalAssessment/analysis/embedding_metrics_and_t-test/bioner_use_t-test-results.json'
    }
}

# 计算相关系数
for task in tasks:
    metric_vecs = {name: load_token_or_embedding_matrix(path, keys) for name, path in metric_paths[task.lower()].items()}
    correlations = compute_correlations(performance_data[task], metric_vecs)
    print(f"Correlations for {task}:")
    for metric, corr in correlations.items():
        print(f"{metric}: {corr}")
    print("\n")
