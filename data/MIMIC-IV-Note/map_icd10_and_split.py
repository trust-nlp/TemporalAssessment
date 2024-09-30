import json
import pandas as pd
from icdmappings import Mapper
from collections import Counter
from transformers import RobertaTokenizer
import os
from sklearn.model_selection import train_test_split

# Initialize Mapper
mapper = Mapper()

# The unmapped file and output path
input_file_path = '/home/weisi/TemporalAssessment/data/MIMIC-IV-Note/mimic.json'
output_file_path = '/home/weisi/TemporalAssessment/data/MIMIC-IV-Note/mimic_final.json'

# Read data into DataFrame
df = pd.read_json(input_file_path, lines=True)

# Fill ICD-10 code by mapping icd9 to icd10
'''def fill_icd10(row):
    if isinstance(row['icd10'], list) and len(row['icd10']) == 0:
        row['icd10'] = mapper.map(row['icd9'], source='icd9', target='icd10')
    return row'''

def fill_icd10(row):
    if isinstance(row['icd10'], list) and len(row['icd10']) == 0:
        # Special case for icd9 code '2720'
        if '2720' in row['icd9']:
            mapped_codes = mapper.map(row['icd9'], source='icd9', target='icd10')
            row['icd10'] = ['E7800' if code == '2720' else other_code for code, other_code in zip(row['icd9'], mapped_codes)]
        else:
            row['icd10'] = mapper.map(row['icd9'], source='icd9', target='icd10')
    return row

df = df.apply(fill_icd10, axis=1)

# Check whether icd10 code is non-empty for all lines
all_entities_have_icd10 = df['icd10'].apply(lambda x: isinstance(x, list) and len(x) > 0).all()
print("ICD-10 code non-empty for all lines:", all_entities_have_icd10)

# Count top frequent 50 icd10 codes
all_icd10_codes = df['icd10'].explode()
#icd10_counter = Counter(all_icd10_codes)
# Exclude 'NoDx'
icd10_counter = Counter(code for code in all_icd10_codes if code != 'NoDx')  
top_50_icd10 = [code for code, count in icd10_counter.most_common(50)]
print("Top frequent 50 ICD-10 codes:", top_50_icd10)


# Generate label column
def generate_label(row):
    row['label'] = [code for code in row['icd10'] if code in top_50_icd10]
    return row

df = df.apply(generate_label, axis=1)

# Truncate text column if " complaint" is found, or remove the standard prefix otherwise
def truncate_text(row):
    text = row['text']
    keyword = "complaint" #usually it's chief complaint, some cases chief is missing
    standard_prefix = "name unit no admission date discharge date date of birth sex service"
    if keyword in text:
        # Split the text at the first occurrence of the keyword and keep the second part
        row['text'] = text.split(keyword, 1)[1]
    elif text.startswith(standard_prefix):
        # Remove the standard prefix if it exists
        row['text'] = text[len(standard_prefix):].lstrip()
    return row

df = df.apply(truncate_text, axis=1)

# Save the result to output file
df.to_json(output_file_path, orient='records', lines=True)

print("Processing finished, file saved in:", output_file_path)

#----------------------------------------
#filter empty label data and short notes 
#----------------------------------------
filtered_df = df[df['label'].apply(lambda x: len(x) > 0)]
print(filtered_df.groupby('time').size())

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def compute_token_length(row):
    tokens = tokenizer.tokenize(row['text'])
    return len(tokens)
filtered_df['token_length'] = filtered_df.apply(compute_token_length, axis=1)

print("Token length statistics:")
'''length_statistics = filtered_df['token_length'].describe()
print(length_statistics)'''

grouped = filtered_df.groupby('time')['token_length'].describe()
print(grouped)
filtered_df['token_length'] = filtered_df.apply(compute_token_length, axis=1)

def count_below_threshold(group, threshold):
    return (group['token_length'] < threshold).sum()

counts_below_256 = filtered_df.groupby('time').apply(count_below_threshold, threshold=256)
counts_below_512 = filtered_df.groupby('time').apply(count_below_threshold, threshold=512)
print("Number of entries with token length below 256 by time:")
print(counts_below_256)
print("\nNumber of entries with token length below 512 by time:")
print(counts_below_512)

print("removed short notes(token_length<256)")
filtered_df=filtered_df[filtered_df['token_length'] > 256]

#----------------------------------------
#split the dataset to 4 time domain 
#----------------------------------------

seed=1
folder_path ='/home/weisi/TemporalAssessment/data/MIMIC-IV-Note/seed{}/'.format(seed)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

ndf=filtered_df[['uid', 'did', 'time', 'text', 'label']]
# devide dataset to 4 time periods
df_2008_2010 = ndf[ndf['time'] == '2008 - 2010']
df_2011_2013 = ndf[ndf['time'] == '2011 - 2013']
df_2014_2016 = ndf[ndf['time'] == '2014 - 2016']
df_2017_2019 = ndf[ndf['time'] == '2017 - 2019']

# reduce the datasets to the same size
min_size = min(len(df_2008_2010), len(df_2011_2013), len(df_2014_2016), len(df_2017_2019))

df_2008_2010_sampled = df_2008_2010.sample(n=min_size, random_state=seed)
df_2011_2013_sampled = df_2011_2013.sample(n=min_size, random_state=seed)
df_2014_2016_sampled = df_2014_2016.sample(n=min_size, random_state=seed)
df_2017_2019_sampled = df_2017_2019.sample(n=min_size, random_state=seed)


def save_datasets(df, period,seed):
    # split train, validation and test datasets by ratio 0.6 0.2 0.2
    print('label list: ',pd.Series(df['label'].explode().unique()))
    train, test = train_test_split(df, test_size=0.4, random_state=seed)  
    validation, test = train_test_split(test, test_size=0.5, random_state=seed)  

    # save files
    train_filename = f'{period}_train.json'
    validation_filename = f'{period}_validation.json'
    test_filename = f'{period}_test.json'
    train.to_json(os.path.join(folder_path, train_filename), orient='records', lines=True)
    validation.to_json(os.path.join(folder_path, validation_filename), orient='records', lines=True)
    test.to_json(os.path.join(folder_path, test_filename), orient='records', lines=True)


save_datasets(df_2008_2010_sampled, 'T1',seed)
save_datasets(df_2011_2013_sampled, 'T2',seed)
save_datasets(df_2014_2016_sampled, 'T3',seed)
save_datasets(df_2017_2019_sampled, 'T4',seed)

print("splited the dataset")