import pandas as pd
import json
import os
from dateutil.parser import parse
from datetime import datetime
from ..dataproc import preprocess
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import gzip

hcpath='/HDD16TB/weisi/amazon_us_reviews/Health_Personal_Care_v1_00/0.1.0/reviews_Health_and_Personal_Care_5.json.gz'

with gzip.open(hcpath,'rt') as f:
    hcdf=pd.read_json(f,lines=True)
    
def convert_to_iso(date_str):
    date_obj = datetime.strptime(date_str, '%m %d, %Y')
    return date_obj.strftime('%Y-%m-%d')

def assign_sentiment(rating):
    if rating in [1, 2]:
        return 'negative'
    elif rating == 3:
        return 'moderate'
    else:  # for ratings 4 and 5
        return 'positive'

print('filling NaN data')    
hcdf.fillna('x', inplace=True) 
print('finish filling')  


hcdf['reviewTime'] = hcdf['reviewTime'].apply(convert_to_iso)

# add a key: year 
hcdf['year'] =  pd.to_datetime(hcdf['reviewTime']).dt.year
# see year distribution:
'''
yearly_counts = hcdf.groupby('year').size()
print(yearly_counts)
'''

# add a key: sentiment
hcdf['sentiment'] = hcdf['overall'].apply(assign_sentiment)



# rename keys if you wish. (model may need to specify text_column_name and label_column_name if it's not the same as the defalt: text/sentence/label...)
# rename overall as label may cause error when running the model(if you want to use sentiment as label, the model automaticly remane it to label but it already exist)
hcdf = hcdf.rename(columns={
    "reviewText": "text",
    "reviewTime": "time"
})

print('cleaning text:') 
tqdm.pandas()#sets up the tqdm progress bar for tracking the progress of the following operation
hcdf.text = hcdf.text.progress_apply(lambda x: preprocess.fast_clean(x))
#join the list of tokens back into a single string with space-separated words.
hcdf.text = hcdf.text.apply(lambda x: ' '.join(x))


# devide dataset to 3 year periods
df_2007_2008 = hcdf[hcdf['year'].isin([2007, 2008])]
df_2009_2010 = hcdf[hcdf['year'].isin([2009, 2010])]
df_2011_2012 = hcdf[hcdf['year'].isin([2011, 2012])]
df_2013_2014 = hcdf[hcdf['year'].isin([2013, 2014])]
df_all_year= hcdf[hcdf['year'].isin([2007, 2014])]

min_size = min(len(df_2007_2008), len(df_2009_2010),len(df_2011_2012), len(df_2013_2014))


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


for seed in range(1, 6):  # randomly split 5 times
    folder_path ='/home/weisi/TemporalAssessment/data/Amazon/HealthCare/seed{}/'.format(seed)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    df_2007_2008_sampled = df_2007_2008.sample(n=min_size, random_state=seed)
    df_2009_2010_sampled = df_2009_2010.sample(n=min_size, random_state=seed)
    df_2011_2012_sampled = df_2011_2012.sample(n=min_size, random_state=seed)
    df_2013_2014_sampled = df_2013_2014.sample(n=min_size, random_state=seed)
    all_year_sampled = df_all_year.sample(n=min_size, random_state=seed)
    split_and_save_datasets(df_2007_2008_sampled, 'amzHC_T1_2007-2008',seed,folder_path)
    split_and_save_datasets(df_2009_2010_sampled, 'amzHC_T2_2009-2010',seed,folder_path)
    split_and_save_datasets(df_2011_2012_sampled, 'amzHC_T3_2011-2012',seed,folder_path)
    split_and_save_datasets(df_2013_2014_sampled, 'amzHC_T4_2013-2014',seed,folder_path)
    split_and_save_datasets(all_year_sampled, 'amzHC_AY_2007-2014',seed,folder_path)



