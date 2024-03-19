import numpy as np
import pandas as pd
import json
import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_hub import KerasLayer


model = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2")
'''model_path = snapshot_download(repo_id="Dimitre/universal-sentence-encoder")
model =  KerasLayer(handle=model_path)'''
#model = pull_from_hub(repo_id="Dimitre/universal-sentence-encoder")
############# NER ##############
'''bioner_path='/home/weisi/TemporalAssessment/data/BioNER/Protein/BIONER-IOBES.json'
df=pd.read_json(bioner_path, lines=True)

df_embeddings = model(df['text'].tolist())
df['use_embedding'] = [embedding.numpy().tolist() for embedding in df_embeddings]
#embedding.tolist() error:  EagerTensor object has no attribute 'tolist'.

# save embedding
output_path = f'/home/weisi/TemporalAssessment/analysis/embeddings/BioNER_embedding-USE.json'
dir_path, filename = os.path.split(output_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
df.to_json(output_path, orient='records', lines=True)
print('bioner embedding saved')'''

########### BioASQ ##########
'''path='/home/weisi/TemporalAssessment/data/BIOASQ/BioASQ.json'
df=pd.read_json(path, lines=True)
#"id"  "type"  
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

df_embeddings = model(df['snippets'].tolist())
df['use_embedding'] = [embedding.numpy().tolist() for embedding in df_embeddings]

df_final = df[['id', 'type', 'use_embedding', 'year', 'domain']]
print('bioasq last sup embedding:',df['use_embedding'].iloc[-1])

# save embedding
output_path = f'/home/weisi/TemporalAssessment/analysis/embeddings/BioASQ_embedding-USE.json'
dir_path, filename = os.path.split(output_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
df_final.to_json(output_path, orient='records', lines=True)
print('bioasq embedding saved')'''
############  MIMIC ###############
mimic_path='/home/weisi/TemporalAssessment/data/MIMIC-IV-Note/mimic-top50.json'
df=pd.read_json(mimic_path, lines=True)
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


def generate_embeddings(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model(batch_texts)
        embeddings.extend(batch_embeddings.numpy().tolist())
    return embeddings

batch_size = 100
df['use_embedding'] = generate_embeddings(df['text'].tolist(), batch_size)


'''df_embeddings = model(df['text'].tolist())
df['use_embedding'] = [embedding.numpy().tolist() for embedding in df_embeddings]'''

print('bioasq last sup embedding:',df['use_embedding'].iloc[-1])
df_final = df[['uid', 'did', 'domain', 'time',  'use_embedding']]

# save embedding
output_path = f'/home/weisi/TemporalAssessment/analysis/embeddings/Mimic_embedding-USE.json'
dir_path, filename = os.path.split(output_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
df_final.to_json(output_path, orient='records', lines=True)
print('mimic embedding saved')




