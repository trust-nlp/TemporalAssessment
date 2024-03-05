from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import json
import os

#model = SentenceTransformer("all-MiniLM-L6-v2")
model_name = "gtr-t5-large"
model = SentenceTransformer(model_name)
# Our sentences we like to encode

path='TemporalAssessment/data/BIOASQ/BioASQ.json'


df=pd.read_json(path, lines=True)
#"id"  "type"  
'''
df_2013_2015 = df[df['year'].isin([2013, 2015])]
df_2016_2018 = df[df['year'].isin([2016, 2018])]
df_2019_2020 = df[df['year'].isin([2019, 2020])]
df_2021_2022 = df[df['year'].isin([2021, 2022])]
'snippets'
"type": "list","summary","yesno","factoid"
'''
#filtered the 2023 data
df= df[df['year'] != 2023]
def assign_domain(year):
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

df['domain'] = df['year'].apply(assign_domain)

embeddings = model.encode(df['snippets'].tolist())
df['embedding'] = list(embeddings)



df_final = df[['id', 'type', 'embedding', 'year', 'domain']]

# save embedding
#output_path = '/home/weisi/TemporalAssessment/analysis/BioASQ_embedding.json'
output_path = f'/home/weisi/TemporalAssessment/analysis/embeddings/BioASQ_embedding-{model_name}.json'
dir_path, filename = os.path.split(output_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
df_final.to_json(output_path, orient='records', lines=True)


domains = ['T1', 'T2', 'T3', 'T4']

# average embedding for each domain
average_embeddings = {}
for domain in domains:
    domain_data = df[df['domain'] == domain]
    average_embedding = np.mean(np.array(domain_data['embedding'].tolist()), axis=0)
    average_embeddings[domain] = average_embedding.tolist()

average_embeddings_path = f'TemporalAssessment/analysis/BioASQ_alltypes_average_embeddings-{model_name}.json'
with open(average_embeddings_path, 'w') as f:
    json.dump(average_embeddings, f)

# cosine simmilarity for each domain pairs 
domain_distances = {}
for i, domain1 in enumerate(domains):
    for domain2 in domains[i+1:]:  # avoid repeated pair
        cos_sim = util.cos_sim(average_embeddings[domain1], average_embeddings[domain2])
        domain_distances[f'{domain1}-{domain2}'] = cos_sim.item()


distances_path = f'/home/weisi/TemporalAssessment/analysis/BioASQ_cosine_alltypes-{model_name}.json'
with open(distances_path, 'w') as f:
    json.dump(domain_distances, f)

