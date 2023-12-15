import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os

df = pd.read_json('/home/weisi/Temporal/data/BIONLP/bionlp.jsonl', lines=True)


df['pmid'] = df['id']
df['id'] = [format(i, 'x') for i in range(len(df))]
df['context'] = df['long']
df['answer'] = df.apply(lambda x: {'text': [x['short']]}, axis=1)

df = df[['id', 'pmid', 'question', 'context', 'answer', 'year']]

year_counts = df.groupby('year').size()
print(year_counts)


# devide dataset to 3 year periods
df_2014_2016 = df[df['year'].isin([2014, 2016])]
df_2017 = df[df['year'] == 2017]
df_2018 = df[df['year']== 2018]
df_2019 = df[df['year']== 2019]
df_2020_2021 = df[df['year'].isin([2020, 2021])]
df_all_year= df[df['year'].isin([2014, 2018])]

min_size = min(len(df_2014_2016), len(df_2017),len(df_2018), len(df_2019))

def split_and_save_datasets(df,period,seed,folder_path):
    # split train, validation and test datasets by ratio 0.6 0.2 0.2
    train, rest = train_test_split(df, test_size=0.4, random_state=seed)  
    validation, test = train_test_split(rest, test_size=0.5, random_state=seed)  
    # save files
    train_filename = f'{period}_train.json'
    validation_filename = f'{period}_validation.json'
    test_filename = f'{period}_test.json'
    train.to_json(os.path.join(folder_path, train_filename), orient='records', lines=True)
    validation.to_json(os.path.join(folder_path, validation_filename), orient='records', lines=True)
    test.to_json(os.path.join(folder_path, test_filename), orient='records', lines=True)


for seed in range(1, 2):  # randomly split 5 times
    folder_path ='/home/weisi/TemporalAssessment/data/BIONLP/newsplit/seed{}/'.format(seed)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    df_2014_2016_sampled = df_2014_2016.sample(n=min_size, random_state=seed)
    df_2017_sampled = df_2017.sample(n=min_size, random_state=seed)
    df_2018_sampled = df_2018.sample(n=min_size, random_state=seed)
    df_2019_sampled = df_2019.sample(n=min_size, random_state=seed)
    all_year_sampled = df_all_year.sample(n=min_size, random_state=seed)
    split_and_save_datasets(df_2014_2016_sampled, 'bionlp_T1_2014-2016',seed,folder_path)
    split_and_save_datasets(df_2017_sampled, 'bionlp_T2_2017',seed,folder_path)
    split_and_save_datasets(df_2018_sampled, 'bionlp_T3_2018',seed,folder_path)
    split_and_save_datasets(df_2019_sampled, 'bionlp_T4_2019',seed,folder_path)
    split_and_save_datasets(all_year_sampled, 'bionlp_AY_2014-2018',seed,folder_path)