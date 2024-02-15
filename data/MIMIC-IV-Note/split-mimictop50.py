import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split


with open('/home/weisi/TemporalAssessment/data/MIMIC-IV-Note/mimic-top50.json', 'r', encoding='utf-8') as f:
    df=pd.read_json(f,lines=True)

# save selected columns
df = df[['uid', 'did', 'time', 'text', 'label']]

folder_path = '/home/weisi/TemporalAssessment/data/MIMIC-IV-Note'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
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



t1 = df[df['time'] == '2008 - 2010']
t2 = df[df['time'] == '2011 - 2013']
t3= df[df['time'] == '2014 - 2016']
t4= df[df['time'] == '2017 - 2019']
df_all_year = pd.concat([t1, t2, t3])

# reduce the datasets to the same size
min_size = min(len(t1), len(t2),len(t3), len(t4))


for seed in range(1, 6):  # randomly split 5 times
    folder_path ='/home/weisi/TemporalAssessment/data/MIMIC-IV-Note/seed{}/'.format(seed)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    t1_sampled = t1.sample(n=min_size, random_state=seed)
    t2_sampled = t2.sample(n=min_size, random_state=seed)
    t3_sampled = t3.sample(n=min_size, random_state=seed)
    t4_sampled = t4.sample(n=min_size, random_state=seed)
    all_year_sampled = df_all_year.sample(n=min_size, random_state=seed)
    split_and_save_datasets(t1_sampled, 'mimic-T1_2008_2010',seed,folder_path)
    split_and_save_datasets(t2_sampled, 'mimic-T2_2011_2013',seed,folder_path)
    split_and_save_datasets(t3_sampled, 'mimic-T3_2014_2016',seed,folder_path)
    split_and_save_datasets(t4_sampled, 'mimic-T4_2017_2019',seed,folder_path)
    split_and_save_datasets(all_year_sampled, 'mimic-AY_2008_2016',seed,folder_path)

