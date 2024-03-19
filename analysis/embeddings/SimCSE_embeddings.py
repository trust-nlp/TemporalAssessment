import numpy as np
import pandas as pd
import json
import os
from simcse import SimCSE

sup_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
unsup_model = SimCSE('princeton-nlp/unsup-simcse-bert-base-uncased')

############# NER ##############
'''bioner_path='/home/weisi/TemporalAssessment/data/BioNER/Protein/BIONER-IOBES.json'
ner=pd.read_json(bioner_path, lines=True)

ner_sup_embeddings = sup_model.encode(ner['text'].tolist())
#type(ner_sup_embeddings)=<class 'torch.Tensor'>
#len(ner_sup_embeddings))=36886 
#Shape of the tensor: torch.Size([36886, 768])
#ner['sup_embedding'] = list(ner_sup_embeddings) will get a list of 36886  torch.Tensor element, json.dumps(ner['sup_embedding'])will fail:OverflowError: Maximum recursion level reached

ner['sup_embedding'] = [embedding.tolist() for embedding in ner_sup_embeddings]
print('last sup embedding:',ner['sup_embedding'].iloc[-1])
embed_as_json = json.dumps(ner['sup_embedding'].iloc[-1])
print(embed_as_json)
ner_unsup_embeddings = unsup_model.encode(ner['text'].tolist())
ner['unsup_embedding'] =  [embedding.tolist() for embedding in ner_unsup_embeddings]

# save embedding
output_path = f'/home/weisi/TemporalAssessment/analysis/embeddings/BioNER_embedding-SimCSE-sup-unsup.json'
dir_path, filename = os.path.split(output_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
ner.to_json(output_path, orient='records', lines=True)
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

sup_embeddings = sup_model.encode(df['snippets'].tolist())
df['sup_embedding'] = [embedding.tolist() for embedding in sup_embeddings]
unsup_embeddings = unsup_model.encode(df['snippets'].tolist())
df['unsup_embedding'] = [embedding.tolist() for embedding in unsup_embeddings]
df_final = df[['id', 'type', 'sup_embedding','unsup_embedding', 'year', 'domain']]
print('bioasq last sup embedding:',df['sup_embedding'].iloc[-1])

# save embedding
output_path = f'/home/weisi/TemporalAssessment/analysis/embeddings/BioASQ_embedding-SimCSE-sup-unsup.json'
dir_path, filename = os.path.split(output_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
df_final.to_json(output_path, orient='records', lines=True)
print('bioasq embedding saved')
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

sup_embeddings = sup_model.encode(df['text'].tolist())
df['sup_embedding'] = [embedding.tolist() for embedding in sup_embeddings]
unsup_embeddings = unsup_model.encode(df['text'].tolist())
df['unsup_embedding'] = [embedding.tolist() for embedding in unsup_embeddings]

print('bioasq last sup embedding:',df['sup_embedding'].iloc[-1])
df_final = df[['uid', 'did', 'domain', 'time',  'sup_embedding','unsup_embedding']]

# save embedding
output_path = f'/home/weisi/TemporalAssessment/analysis/embeddings/Mimic_embedding-SimCSE-sup-unsup.json'
dir_path, filename = os.path.split(output_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
df_final.to_json(output_path, orient='records', lines=True)
print('mimic embedding saved')
