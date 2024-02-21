import pandas as pd
from sklearn.model_selection import train_test_split
import os

alldf = pd.read_json('/home/weisi/TemporalAssessment/data/BIOASQ/BioASQ.json', lines=True)

# split the yesno questions data
df=alldf[alldf['type']=='list']
df_2013_2015 = df[df['year'].isin([2013, 2015])]
df_2016_2018 = df[df['year'].isin([2016, 2018])]
df_2019_2020 = df[df['year'].isin([2019, 2020])]
df_2021_2022 = df[df['year'].isin([2021, 2022])]
df_all_year= df[df['year'].isin([2013, 2020])]

min_size = min(len(df_2013_2015), len(df_2016_2018),len(df_2019_2020), len(df_2021_2022))

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

for seed in range(1, 6):  # range(1,6):randomly split 5 times
    folder_path ='/home/weisi/TemporalAssessment/data/BIOASQ/list/seed{}/'.format(seed)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    df_2013_2015_sampled = df_2013_2015.sample(n=min_size, random_state=seed)
    df_2016_2018_sampled = df_2016_2018.sample(n=min_size, random_state=seed)
    df_2019_2020_sampled = df_2019_2020.sample(n=min_size, random_state=seed)
    df_2021_2022_sampled = df_2021_2022.sample(n=min_size, random_state=seed)
    all_year_sampled = df_all_year.sample(n=min_size, random_state=seed)
    split_and_save_datasets(df_2013_2015_sampled, 'bioasq-list-T1_2013_2015',seed,folder_path)
    split_and_save_datasets(df_2016_2018_sampled, 'bioasq-list-T2_2016_2018',seed,folder_path)
    split_and_save_datasets(df_2019_2020_sampled, 'bioasq-list-T3_2019_2020',seed,folder_path)
    split_and_save_datasets(df_2021_2022_sampled, 'bioasq-list-T4_2021_2022',seed,folder_path)
    split_and_save_datasets(all_year_sampled, 'bioasq-list-AY_2013_2020',seed,folder_path)
