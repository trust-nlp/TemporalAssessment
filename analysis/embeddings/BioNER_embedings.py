from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import json
import os

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Our sentences we like to encode

bioner_path='/home/weisi/TemporalAssessment/data/BioNER/Protein/BIONER-IOBES.json'


ner=pd.read_json(bioner_path, lines=True)


# text are encoded by calling model.encode()
#ner['embedding']=model.encode(ner['text'])

embeddings = model.encode(ner['text'].tolist())

ner['embedding'] = list(embeddings)

# save embedding
output_path = f'/home/weisi/TemporalAssessment/analysis/embeddings/BioNER_embedding-{model_name}.json'
dir_path, filename = os.path.split(output_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

ner.to_json(output_path, orient='records', lines=True)


domains = ['T1', 'T2', 'T3', 'T4']


# average embedding for each domain
average_embeddings = {}
for domain in domains:
    domain_data = ner[ner['domain'] == domain]
    average_embedding = np.mean(np.array(domain_data['embedding'].tolist()), axis=0)
    average_embeddings[domain] = average_embedding.tolist()

average_embeddings_path = f'/home/weisi/TemporalAssessment/analysis/BioNER_average_embeddings-{model_name}.json'

with open(average_embeddings_path, 'w') as f:
    json.dump(average_embeddings, f)
'''
# cosine simmilarity for each domain pairs 
domain_distances = {}
for i, domain1 in enumerate(domains):
    for domain2 in domains[i+1:]:  # avoid repeated pair
        cos_sim = util.cos_sim(average_embeddings[domain1], average_embeddings[domain2])
        domain_distances[f'{domain1}-{domain2}'] = cos_sim.item()


distances_path = f'/home/weisi/TemporalAssessment/analysis/BioNER_cosine-{model_name}.json'

with open(distances_path, 'w') as f:
    json.dump(domain_distances, f)'''
