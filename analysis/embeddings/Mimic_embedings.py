from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import json
import os

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
# Our sentences we like to encode

path='/home/weisi/TemporalAssessment/data/MIMIC-IV-Note/mimic-top50.json'
df=pd.read_json(path, lines=True)

'''/home/weisi/TemporalAssessment/data/MIMIC-IV-Note/mimic-top50.json
'time'=
'2008 - 2010'
'2011 - 2013'
'2014 - 2016'
'2017 - 2019'
text
'''
#filter data
df= df[df['time'] != '2020 - 2022' ]

def assign_domain(time):
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

df['domain'] = df['time'].apply(assign_domain)

embeddings = model.encode(df['text'].tolist())
df['embedding'] = list(embeddings)


df_final = df[['uid', 'did', 'domain', 'time', 'embedding']]

# save embedding
output_path = f'/home/weisi/TemporalAssessment/analysis/embeddings/Mimic_embedding-{model_name}.json'
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

average_embeddings_path = f'/home/weisi/TemporalAssessment/analysis/Mimic_average_embeddings-{model_name}.json'
with open(average_embeddings_path, 'w') as f:
    json.dump(average_embeddings, f)

# cosine simmilarity for each domain pairs 
domain_distances = {}
for i, domain1 in enumerate(domains):
    for domain2 in domains[i+1:]:  # avoid repeated pair
        cos_sim = util.cos_sim(average_embeddings[domain1], average_embeddings[domain2])
        domain_distances[f'{domain1}-{domain2}'] = cos_sim.item()


distances_path = f'/home/weisi/TemporalAssessment/analysis/Mimic_cosine-{model_name}.json'
with open(distances_path, 'w') as f:
    json.dump(domain_distances, f)

