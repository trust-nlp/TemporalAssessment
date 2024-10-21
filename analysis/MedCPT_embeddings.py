import numpy as np
import pandas as pd
import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "ncbi/MedCPT-Article-Encoder"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)  # 发送模型到GPU


def calculate_embeddings(texts):
    with torch.no_grad():
        # Tokenize the texts
        encoded = tokenizer(texts, truncation=True, padding=True, return_tensors='pt', max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        # Obtain the outputs from the model
        outputs = model(**encoded)
        # Get the [CLS] token's embeddings (representative of the whole input)
        #embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = outputs.last_hidden_state[:, 0, :] # 将嵌入结果送回CPU
    
    return embeddings.cpu().numpy()  # Move embeddings to CPU and convert to numpy

def batch_process(texts, batch_size=10):
    batched_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i+batch_size]
        embeddings = calculate_embeddings(batch_texts)
        batched_embeddings.extend(embeddings)
    return np.array(batched_embeddings)

############# NER ##############
bioner_path='/home/weisi/TemporalAssessment/data/BioNER/Protein/BIONER-IOBES.json'
df=pd.read_json(bioner_path, lines=True)
embeddings = batch_process(df['text'].tolist(), batch_size=100)
df['embedding'] = list(embeddings)


# save embedding
output_path = f'/home/weisi/TemporalAssessment/analysis/embeddings/BioNER_embedding-MedCPT.json'
dir_path, filename = os.path.split(output_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
df.to_json(output_path, orient='records', lines=True)
print('bioner embedding saved')
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
embeddings = batch_process(df['snippets'].tolist(), batch_size=100)
df['embedding'] = list(embeddings)

df_final = df[['id', 'type', 'embedding', 'year', 'domain']]


# save embedding
output_path = f'/home/weisi/TemporalAssessment/analysis/embeddings/BioASQ_embedding-MedCPT.json'
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

# Downsampling
'''seed = 1
time_periods = ['2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019']
dfs = []
min_size = min(df[df['time'] == period].shape[0] for period in time_periods)

for period in time_periods:
    dfs.append(df[df['time'] == period].sample(n=min_size, random_state=seed))

df_downsampled = pd.concat(dfs)'''

# Assign domain
def assign_domain(time):
    if time == '2008 - 2010':
        return 'T1'
    elif time == '2011 - 2013':
        return 'T2'
    elif time == '2014 - 2016':
        return 'T3'
    elif time == '2017 - 2019':
        return 'T4'
    else:
        raise ValueError(f'time {time} does not match any domain criteria')

df['domain'] = df['time'].apply(assign_domain)

# Calculate embeddings in batches
embeddings = batch_process(df['text'].tolist(), batch_size=100)
df['embedding'] = list(embeddings)

# Prepare final dataframe
df_final = df[['uid', 'did', 'domain', 'time', 'embedding']]


# save embedding
output_path = f'/home/weisi/TemporalAssessment/analysis/embeddings/Mimic_embedding-MedCPT.json'
dir_path, filename = os.path.split(output_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
df_final.to_json(output_path, orient='records', lines=True)
print('mimic embedding saved')




