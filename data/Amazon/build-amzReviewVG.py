import pandas as pd
import json
import os
from collections import Counter
import re
from dateutil.parser import parse
import xmltodict
import yaml
import sys
import glob
import ast
from datetime import datetime
from dataproc import preprocess
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import gzip

# review data of Video Game

# v2 2018 dataset: (request via google form is required , see https://nijianmo.github.io/amazon/#complete-data)
# wget is not available
# v1 2014 dataset
# wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games.json.gz 
def convert_to_iso(date_str):
    date_obj = datetime.strptime(date_str, '%m %d, %Y')
    return date_obj.strftime('%Y-%m-%d')

vgpath='/HDD16TB/weisi/Video_Games.json.gz'

with gzip.open(vgpath,'rt') as f:
    vgdf=pd.read_json(f,lines=True)

print('filling NaN data')    
vgdf.fillna('x', inplace=True) 
print('finish filling')  

vgdf['reviewTime'] = vgdf['reviewTime'].apply(convert_to_iso)

# add a key: year 
vgdf['year'] =  pd.to_datetime(vgdf['reviewTime']).dt.year
yearly_counts = vgdf.groupby('year').size()
# rename keys
vgdf = vgdf.rename(columns={
    "overall": "label",
    "reviewText": "text",
    "reviewTime": "time"
})

vgdf_2001 = vgdf[vgdf['year'] == 2001]

print('cleaning text:') 
tqdm.pandas()#sets up the tqdm progress bar for tracking the progress of the following operation
vgdf_2001.text = vgdf_2001.text.progress_apply(lambda x: preprocess.fast_clean(x))
#join the list of tokens back into a single string with space-separated words.
vgdf_2001.text = vgdf_2001.text.apply(lambda x: ' '.join(x))
vgdf_2001.to_json('amazonVG_2001.json', orient='records', lines=True)

