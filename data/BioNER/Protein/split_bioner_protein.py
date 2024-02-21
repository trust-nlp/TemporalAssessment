import pandas as pd
from sklearn.model_selection import train_test_split
import os

basepath='/home/weisi/TemporalAssessment/data/BioNER/Protein'
'''prefixes = [
    'BioNLP09-IOB',
    'BioNLP09-IOBES',
    'BioNLP11EPI-IOB',
    'BioNLP11EPI-IOBES',
    'BioNLP11ID-IOB',
    'BioNLP11ID-IOBES',
    'BioNLP13GE-IOB',
    'BioNLP13GE-IOBES'
]'''

d1_09_iobes=pd.read_json(os.path.join(basepath,'BioNLP09-IOBES.json'), lines=True)
d2_11epi_iobes=pd.read_json(os.path.join(basepath,'BioNLP11EPI-IOBES.json'), lines=True)
d3_11id_iobes=pd.read_json(os.path.join(basepath,'BioNLP11ID-IOBES.json'), lines=True)
d4_13ge_iobes=pd.read_json(os.path.join(basepath,'BioNLP13GE-IOBES.json'), lines=True)

#min_size = min(len(d1_09_iobes),len(d2_11epi_iobes),len(d3_11id_iobes),len(d4_13ge_iobes))

def split_and_save_datasets(df,period,seed,folder_path):
    # split train, validation and test datasets by ratio 0.6 0.2 0.2
    train, rest = train_test_split(df, test_size=0.4, random_state=seed)  
    validation, test = train_test_split(rest, test_size=0.5, random_state=seed)  
    # save files
    train_filename = f'{period}-train.json'
    validation_filename = f'{period}-validation.json'
    test_filename = f'{period}-test.json'
    train.to_json(os.path.join(folder_path, train_filename), orient='records', lines=True)
    validation.to_json(os.path.join(folder_path, validation_filename), orient='records', lines=True)
    test.to_json(os.path.join(folder_path, test_filename), orient='records', lines=True)


def add_entity_count(df):
    df['entity_count'] = df['ner_tags'].apply(lambda tags: tags.count("B-Protein") + tags.count("S-Protein"))

for df in [d1_09_iobes, d2_11epi_iobes, d3_11id_iobes, d4_13ge_iobes]:
    add_entity_count(df)

min_entity_count = min(df['entity_count'].sum() for df in [d1_09_iobes, d2_11epi_iobes, d3_11id_iobes, d4_13ge_iobes])
print('min_entity_count=',min_entity_count)
def sample_to_target_entities(df, target_entity_count, seed, max_iter=100, tolerance=0.001):
    estimated_sample_size = int(target_entity_count / df['entity_count'].mean())
    for i in range(max_iter):
        #sampled_df = df.sample(n=min(len(df), estimated_sample_size), weights='entity_count', random_state=seed)
        sampled_df = df.sample(n=min(len(df), estimated_sample_size), random_state=seed)
        sampled_entity_count = sampled_df['entity_count'].sum()

        diff_ratio = (sampled_entity_count - target_entity_count) / target_entity_count

        if abs(diff_ratio) <= tolerance:  
            print("succeed in", i,'th iteration, diff=', sampled_entity_count - target_entity_count)
            return sampled_df
        
        # adjust sample size if the diff is not tolerated
        adjustment_factor = 1 - diff_ratio 
        estimated_sample_size = int(estimated_sample_size * adjustment_factor)

    return None 


for seed in range(1, 6):  # range(1,6):randomly split 5 times
    folder_path ='/home/weisi/TemporalAssessment/data/BioNER/Protein/seed{}/'.format(seed)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    #df_sampled = df.sample(n=min_size, random_state=seed)
    for df, prefix in zip([d1_09_iobes, d2_11epi_iobes, d3_11id_iobes, d4_13ge_iobes], ['D1_09', 'D2_11EPI', 'D3_11ID', 'D4_13GE']):
        sampled_df = sample_to_target_entities(df, min_entity_count, seed)
        if sampled_df is not None:
            split_and_save_datasets(sampled_df, f'{prefix}_iobes', seed, folder_path)
        
    '''
    split_and_save_datasets(d1_09_iobes, 'D1_09_unsampled_iobes',seed,folder_path)
    split_and_save_datasets(d2_11epi_iobes, 'D2_11EPI_unsampled_iobes',seed,folder_path)
    split_and_save_datasets(d3_11id_iobes, 'D3_11ID_unsampled_iobes',seed,folder_path)
    split_and_save_datasets(d4_13ge_iobes, 'D4_13GE_unsampled_iobes',seed,folder_path)'''
    
