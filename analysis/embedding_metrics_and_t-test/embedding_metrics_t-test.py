
'''Steps:
1. compute 15 random result of average embeddings for each domain, extract half of the domain data each time
    save it to random_average_embeddings_each_domain.json
2. for each domain pair, compute result on 3 metrics: cosine, manhattan, Euclidean,
    save them as cosine_list, manhattan_list and Euclidean_list
3. Randomly split the all time domain data to 2 halves,   
    compute 15 random result of average embeddings for each half, 
    and compute the cosine, manhattan, Euclidean between the 2 half average embeddings each time, 
    save it as the base_cosine list, base_manhattan list and base_Euclidean list
    and same operations for each domain data.
4. conduct t-test between metric_list and base_metric_list for each domain pairs and each metric

data to save:
domian Ti:
    average_embedding_list_half1: (length=15)
    average_embedding_list_half2: (length=15)
the whole dataset:
    average_embedding_list_half1: (length=15)
    average_embedding_list_half2: (length=15)
    base_cosine_list, base_manhattan_list, base_Euclidean_list. 
    computed from average_embedding_list_half1 and average_embedding_list_half2)
domain pair Ti_Tj: (including i=j)
    cosine_list, manhattan_list, Euclidean_list. 
    (computed from average_embedding_list_half1 of Ti and average_embedding_list_half2 of Tj)

T-test results:   
1. two tailed T-test result of metric_list of each domain pair Ti_Tj vs base_metric_list. p-value and whether null hyposis is reject
(null hypothesis is the mean value of metric_list is equal to that of base_metric_list)
2. single-tail One-sample t-test T-test result: result of metric_list of each domain pair Ti_Tj vs 0 or 1 
(for cosine: null hypothesis is the mean value of metric_list is equal to 1, which means the cosine similarity is significant
for distances: null hypothesis is the mean value of metric_list is equal to 0, which means the distance is not significant)
'''

'''our method: previous research only use cosine similarity for embedding, 
 the way we conduct t-test, the concept of base_metric
 and we will have 15*10(10 domain pairs) of data point for each metric and we have 5 seeds of performance table (5*10 Y)
so we can do regression and see the correlation of metrics and performance'''
import numpy as np
import os
import json
import pandas as pd
#from scipy.spatial.distance import  euclidean
from scipy.stats import ttest_1samp, ttest_ind
from sklearn.metrics.pairwise import paired_distances #cosine_similarity, manhattan_distances

#dataframes:
bioasq_gtr_t5=pd.read_json('/home/weisi/TemporalAssessment/analysis/embeddings/BioASQ_embedding-gtr-t5-large.json', lines=True)
bioasq_miniLM=pd.read_json('/home/weisi/TemporalAssessment/analysis/embeddings/BioASQ_embedding-all-MiniLM-L6-v2.json', lines=True)
mimic_miniLM=pd.read_json('/home/weisi/TemporalAssessment/analysis/embeddings/Mimic_embedding-all-MiniLM-L6-v2.json', lines=True)
bioner_miniLM=pd.read_json('/home/weisi/TemporalAssessment/analysis/embeddings/BioNER_embedding-all-MiniLM-L6-v2.json', lines=True)

df_list = [bioasq_gtr_t5, bioasq_miniLM, mimic_miniLM, bioner_miniLM]
domains = ['T1', 'T2', 'T3', 'T4']
# embeddings: sbert first 512, sbert average sentence/paragragh, or universal sentence embedding(USE), or SimCSE, or average token embedding
# metrics: cosine, manhattan, Euclidean,


def compute_average_embedding(df, sample_ratio=0.5):
    sample_size = int(len(df) * sample_ratio)
    sampled_df = df.sample(n=sample_size)
    average_embedding = np.mean(np.stack(sampled_df['embedding']), axis=0)
    return average_embedding

def compute_base_average_embeddings_list(df, num_iterations=15):
    results_half1 = []
    results_half2 = []
    for _ in range(num_iterations):
        sampled_df = df.sample(frac=0.5)
        average_embedding_half1 = np.mean(np.stack(sampled_df['embedding']), axis=0)
        results_half1.append(average_embedding_half1.tolist())
    
        remaining_df = df.drop(sampled_df.index)
        average_embedding_half2 = np.mean(np.stack(remaining_df['embedding']), axis=0)
        results_half2.append(average_embedding_half2.tolist())
    return results_half1, results_half2

def compute_domain_average_embeddings_list(df, domain, num_iterations=15):
    results_half1 = []
    results_half2 = []
    for _ in range(num_iterations):
        sampled_df = df[df['domain'] == domain].sample(frac=0.5)
        average_embedding_half1 = np.mean(np.stack(sampled_df['embedding']), axis=0)
        results_half1.append(average_embedding_half1.tolist())
    
        remaining_df = df[df['domain'] == domain].drop(sampled_df.index)
        average_embedding_half2 = np.mean(np.stack(remaining_df['embedding']), axis=0)
        results_half2.append(average_embedding_half2.tolist())
    return results_half1, results_half2

def compute_metric_lists(results_half1, results_half2):
    # make sure the input is np array
    results_half1_np = np.array(results_half1)
    results_half2_np = np.array(results_half2)
    
    euclidean_dists = paired_distances(results_half1_np, results_half2_np, metric='euclidean')
    manhattan_dists = paired_distances(results_half1_np, results_half2_np, metric='manhattan')
    cosine_dists = paired_distances(results_half1_np, results_half2_np, metric='cosine')
    cosine_sims = 1 - cosine_dists
    metrics = {
        "cosine_similarity": cosine_sims.tolist(),
        "cosine_distance": cosine_dists.tolist(),
        "manhattan_distance": manhattan_dists.tolist(),
        "euclidean_distance": euclidean_dists.tolist(),
    }
    return metrics


def conduct_t_tests(metrics, base_metrics):
    t_test_results = {}
    for metric in metrics.keys():
        '''1. Two tailed T-test result of metric_list of each domain pair Ti_Tj vs base_metric_list. p-value and whether null hyposis is reject
            (null hypothesis is the mean value of metric_list is equal to that of base_metric_list)
        '''
        t_stat, p_val = ttest_ind(metrics[metric], base_metrics[metric])
        t_test_results[metric] = {"against_base": {"t_stat": t_stat, "p_val": p_val}}
        '''2. single-tail One-sample t-test T-test result: result of metric_list of each domain pair Ti_Tj vs 0 or 1 
        '''
        if metric in ["cosine_similarity"]:
            popmean = 1
        else:
            popmean = 0
        t_stat, p_val = ttest_1samp(metrics[metric], popmean=popmean)
        t_test_results[metric]["against_constant"] = {"t_stat": t_stat, "p_val": p_val}
    return t_test_results

def conduct_t_tests_Welch(metrics, base_metrics):
    t_test_results = {}
    for metric in metrics.keys():
        t_stat, p_val = ttest_ind(metrics[metric], base_metrics[metric], equal_var=False)
        t_test_results[metric] = {"against_base": {"t_stat": t_stat, "p_val": p_val}}
        
        if metric in ["cosine_similarity"]:
            popmean = 1
        else:
            popmean = 0
        t_stat, p_val = ttest_1samp(metrics[metric], popmean=popmean)
        t_test_results[metric]["against_constant"] = {"t_stat": t_stat, "p_val": p_val}
    return t_test_results




for df_index, df in enumerate(df_list):
    df_results = {}
    df_all_results = {}
    #df_name = f"DataFrame_{df_index}"
    #df_results[df_name] = {}
    
    base_ave_emb_half1_list, base_ave_emb_half2_list = compute_base_average_embeddings_list(df)
    base_metrics = compute_metric_lists(base_ave_emb_half1_list, base_ave_emb_half2_list)
    
    metrics_avg = {}
    for metric, values in base_metrics.items():
        metrics_avg[metric] = np.mean(values)

    df_results["base_metrics"] = metrics_avg
    df_all_results["base_metrics"] = base_metrics

    domain_results = {}
    for i, domain1 in enumerate(domains):
        for domain2 in domains[i:]:
            emb1_half1_list, emb1_half2_list = compute_domain_average_embeddings_list(df, domain1)
            emb2_half1_list, emb2_half2_list = compute_domain_average_embeddings_list(df, domain2)
            metrics = compute_metric_lists(emb1_half1_list, emb2_half2_list)
            t_test_results = conduct_t_tests(metrics, base_metrics)
            metrics_avg = {}
            for metric, values in metrics.items():
                metrics_avg[metric] = np.mean(values)

            domain_pair_key = f"{domain1}_{domain2}"
            df_results[domain_pair_key] = {"metrics": metrics_avg, "t_test_results": t_test_results}
            df_all_results[domain_pair_key] = {"metrics": metrics}

    
    output_path1 = f'/home/weisi/TemporalAssessment/analysis/t-test/{df_index}_t-test-results.json'
    dir_path, filename = os.path.split(output_path1)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(output_path1, 'w') as f:
        json.dump(df_results, f, indent=4)
    
    output_path2 = f'/home/weisi/TemporalAssessment/analysis/t-test/{df_index}_metric_lists.json'
    dir_path, filename = os.path.split(output_path2)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(output_path2, 'w') as f:
        json.dump(df_all_results, f, indent=4)