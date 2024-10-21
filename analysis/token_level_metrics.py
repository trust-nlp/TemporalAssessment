'''
1. get the 1 gram or  1-3 gram token dicts and word dicts(excluede english stop words) of 3 datatsets, rank and save, 
    see most 20 frequent and least frequent tokens and words
2. calculate jaccard similarity of each domain pair
3. select the most 10% frequent tokens of the whole dataset, lets' say there are N tokens for each
    for each time domain, calculate tf-idf for the most 10% frequent 10% frequent tokens, get 2 vectors of length N
    and for each domain pair ,compute the cosine of for the most 10% frequent tf-idf vectors of 2 domains tf-idf vectors  of 2 domains;
    Note:  least 10% frequent tokens will fail because the vector has extremly small values and norm will be 0, which casuse nan in cosine

other possible metric: get the most 10% frequent and least frequent tokens of each time domain, combine them to a sentence, 
        compute word mover distance of each domain pair

T-TEST seems not suitable
'''
import json
import os
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Make sure to download these resources if you haven't already
'''nltk.download('punkt')
nltk.download('stopwords')'''

bioasq_path='/home/weisi/TemporalAssessment/data/BIOASQ/BioASQ_formatted.json'
bioner_path='/home/weisi/TemporalAssessment/data/BioNER/Protein/BIONER-IOBES.json'
mimic_path='/home/weisi/TemporalAssessment/data/MIMIC-IV-Note/mimic-top50.json'
domains = ['T1', 'T2', 'T3', 'T4']

bioner=pd.read_json(bioner_path, lines=True)
bioasq=pd.read_json(bioasq_path, lines=True)
mimic=pd.read_json(mimic_path, lines=True)



mimic= mimic[mimic['time'] != '2020 - 2022' ]
def assign_domain_mimic(time):
    if time =='2008 - 2010':
        return 'T1'
    elif time in '2011 - 2013':
        return 'T2'
    elif time in '2014 - 2016':
        return 'T3'
    elif time in '2017 - 2019':
        return 'T4'
    else:
        raise ValueError(f'time {time} does not match any domain criteria')
mimic['domain'] = mimic['time'].apply(assign_domain_mimic)

bioasq= bioasq[bioasq['year'] != 2023]
def assign_domain_bioasq(year):
    if year in [2013,2014,2015]:
        return 'T1'
    elif year in [2016,2017,2018]:
        return 'T2'
    elif year in [2019, 2020]:
        return 'T3'
    elif year in [2021, 2022]:
        return 'T4'
    else:
        raise ValueError(f'Year {year} does not match any domain criteria')

bioasq['domain'] = bioasq['year'].apply(assign_domain_bioasq)
bioasq['text']=bioasq['snippets']

#df_list=[bioasq,mimic,bioner]
df_dict = {
    'bioasq': bioasq, 
    'mimic': mimic, 
    'bioner': bioner
}

#-------------------------


stop_words = set(stopwords.words('english'))

def tokenize_and_count_ngrams(texts, n_range=(1, 3)):
    """
    Tokenize texts and generate n-grams within the specified range, excluding stopwords.
    Returns Counter object with frequencies of ONE gram, n-gram, and all token set(1-gram).
    """
    #word_counter = Counter()
    all_ngrams = Counter()
    one_gram_tokens = Counter()
    all_tokens_set = set() 
    #ngram_tokens_set = set() 
    for text in texts:
        # Tokenization and stopword removal
        tokens = [token for token in nltk.word_tokenize(text.lower()) if token.isalpha() and token not in stop_words]
        all_tokens_set.update(tokens) 
        one_gram_tokens.update(tokens) 
        # Generating n-grams for the specified range and counting
        for n in range(n_range[0], n_range[1] + 1):
            n_grams = ngrams(tokens, n)
            all_ngrams.update(n_grams)
    
    return one_gram_tokens,all_ngrams,all_tokens_set

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


# Assuming domain_tokens is a dict with domain as keys and sets of tokens as values

# Prepare domain_texts for each domain in a dist : {'T1': ['text1', 'text2', ...], 'T2': [...], ...}
#for df_index, df in enumerate(df_list):
for df_name, df in df_dict.items():
    domain_texts = {domain: [] for domain in domains}
    
    for _, row in df.iterrows():
        domain_texts[row['domain']].append(row['text'])

    # domain_tokens = {'T1': set([...]), 'T2': set([...]), ...}
    domain_tokens_sets = {domain: set() for domain in domains}  
    for domain in domains:
        # Process texts for each domain separately
        _, _, tokens_set = tokenize_and_count_ngrams(domain_texts[domain])
        domain_tokens_sets[domain] = tokens_set

    jaccard_results = {}
    for i, domain1 in enumerate(domains):
        for domain2 in domains[i:]: # To ensure each pair is calculated once
            domain_pair_key = f"{domain1}_{domain2}"
            sim = jaccard_similarity(domain_tokens_sets[domain1], domain_tokens_sets[domain2])
            jaccard_results[domain_pair_key] = sim
            #jaccard_results[(domain1, domain2)] = sim
    print(df_name,':',jaccard_results,'\n')
    output_path1 = f'/home/weisi/TemporalAssessment/analysis/token_metrics/{df_name}_jaccard_results.json'
    dir_path, filename = os.path.split(output_path1)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(output_path1, 'w') as f:
        json.dump(jaccard_results, f, indent=4)

    ##--------------tfidf cosine----------------------

    texts_df = df['text'].tolist()  # may need to Replace 'text' with the actual text column name
    one_grams,_,tokens = tokenize_and_count_ngrams(texts_df)
    # one_grams is a Counter object of tokens from whole datasets
    total_tokens = len(one_grams)
    most_frequent_tokens = one_grams.most_common(int(total_tokens * 0.1))
    # Convert to sets or lists as needed
    most_frequent_tokens_set = set([token for token, freq in most_frequent_tokens])
    # Calculate TF-IDF for the most and least frequent tokens
    vectorizer_most = TfidfVectorizer(vocabulary=most_frequent_tokens_set)
    # domain_texts is a dist : {'T1': ['text1', 'text2', ...], 'T2': [...], ...}
    tfidf_vectors_most = {domain: vectorizer_most.fit_transform(domain_texts[domain]).mean(axis=0).A1 for domain in domains}
    '''tf_idf_vec=tfidf_matrix.mean(axis=0).A1'''
    # Calculate cosine similarity between domain pairs
    cosine_similarities_most = {}


    for i, domain1 in enumerate(domains):
        for domain2 in domains[i:]:
            domain_pair_key = f"{domain1}_{domain2}"
            most_vec1=tfidf_vectors_most[domain1]
            most_vec2=tfidf_vectors_most[domain2]
            sim_most = np.dot(most_vec1, most_vec2)/(np.linalg.norm(most_vec1)*np.linalg.norm(most_vec2))
            cosine_similarities_most[domain_pair_key] = sim_most

    # Save or print the cosine similarity results
    print(df_name,':',"Cosine Similarities for Most Frequent Tokens:", cosine_similarities_most,'\n')
    output_path2 = f'/home/weisi/TemporalAssessment/analysis/token_metrics/{df_name}_cosine_tfidf_of_most_freq_tokens.json'
    dir_path, filename = os.path.split(output_path2)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(output_path2, 'w') as f:
        json.dump(cosine_similarities_most, f, indent=4)

