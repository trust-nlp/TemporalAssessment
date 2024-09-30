from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import json
import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_hub import KerasLayer


model_name = 'FremyCompany/BioLORD-2023'
model = SentenceTransformer(model_name)

############# NER ##############
'''bioner_path='/home/weisi/TemporalAssessment/data/BioNER/Protein/BIONER-IOBES.json'
df=pd.read_json(bioner_path, lines=True)

#embedding.tolist() error:  EagerTensor object has no attribute 'tolist'.
embeddings = model.encode(df['text'].tolist())
df['embedding'] = list(embeddings)

# save embedding
output_path = f'/home/weisi/TemporalAssessment/analysis/embeddings/BioNER_embedding-BioLORD.json'
dir_path, filename = os.path.split(output_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
df.to_json(output_path, orient='records', lines=True)
print('bioner embedding saved')'''

########### BioASQ ##########
path='/home/weisi/TemporalAssessment/data/BIOASQ/BioASQ.json'
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


embeddings = model.encode(df['snippets'].tolist())
df['embedding'] = list(embeddings)

df_final = df[['id', 'type', 'embedding', 'year', 'domain']]


# save embedding
output_path = f'/home/weisi/TemporalAssessment/analysis/embeddings/BioASQ_embedding-BioLORD.json'
dir_path, filename = os.path.split(output_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
df_final.to_json(output_path, orient='records', lines=True)
print('bioasq embedding saved')

############  MIMIC ###############
mimic_path='/home/weisi/TemporalAssessment/data/MIMIC-IV-Note/mimic_final.json'

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


embeddings = model.encode(df['text'].tolist())
df['embedding'] = list(embeddings)


df_final = df[['uid', 'did', 'domain', 'time', 'embedding']]


# save embedding
output_path = f'/home/weisi/TemporalAssessment/analysis/embeddings/Mimic_embedding-BioLORD.json'
dir_path, filename = os.path.split(output_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
df_final.to_json(output_path, orient='records', lines=True)
print('mimic embedding saved')




