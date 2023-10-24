import pandas as pd
import json
import os
from collections import Counter
import re
from dateutil.parser import parse

from datetime import datetime
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import gzip

path='/HDD16TB/weisi/amazon_us_reviews/Health_Personal_Care_v1_00/0.1.0/reviews_Health_and_Personal_Care_5.json.gz'

with gzip.open(path,'rt') as f:
    df=pd.read_json(f,lines=True)

tok = RegexpTokenizer(r'\w+')  # alphanumeric tokenization

def preprocess(doc, stopwords=None):
    global tok
    # If the 'stopwords' argument is not provided, set it to an empty set.
    if stopwords is None:
        stopwords = set()
    # replace URLs in the doc string with the word "url"
    doc = re.sub(r"https?:\S+", "url", doc)
    # remove newlines and tabs 
    doc = doc.replace('\n', ' ')
    doc = doc.replace('\t', ' ')
    # replace date
    doc = re.sub(r"(\d+)+(\-)+(\d+)+(\-)+(\d+)", "date", doc)
    # remove all serialization eg 1. 1) or 1.1
    doc = re.sub(r"(\d+)+(\.|\))+(\d+)", "", doc)
    doc = re.sub(r"(\d+)+(\.|\))", "", doc)
    doc = re.sub(r"\b(\w+)( \1\b)+", r"\1", doc)  # removing consecutive duplicate words
    doc = re.sub(r"([^A-Za-z0-9\s](\s)){2,}", " ", doc)  # remove consecutive punctuations
    doc = re.sub(r'\.+', '..', doc) #replaces two or more consecutive ellipsis with just two (..).
    doc = re.sub(r'!+', '!', doc) #replaces two or more consecutive ! with just one 
    doc = re.sub(r'\*+', ' ', doc) #replaces two or more consecutive * with just space
    doc = re.sub(r'_+', ' ', doc) #replaces two or more consecutive underscore_
    doc = re.sub(r',+', ',', doc) #replaces two or more consecutive commas,
    # all lowercase
    doc = doc.lower()
    doc = [item.strip() for item in tok.tokenize(doc)
           if len(item.strip()) > 1 and item not in stopwords
           ]  # tokenize
    return doc

tqdm.pandas()#sets up the tqdm progress bar for tracking the progress of the following operation
df.reviewText = df.reviewText.progress_apply(lambda x: preprocess(x))

#join the list of tokens back into a single string with space-separated words.
df.reviewText = df.reviewText.apply(lambda x: ' '.join(x))

def convert_to_iso(date_str):
    date_obj = datetime.strptime(date_str, '%m %d, %Y')
    return date_obj.strftime('%Y-%m-%d')

df['reviewTime'] = df['reviewTime'].apply(convert_to_iso)

# Rename columns
df = df.rename(columns={
    "reviewerID": "uid",
    "reviewText": "text",
    "reviewTime": "time"
})

# Write the dataframe to a JSON file
with open('amazon.json', 'w') as wfile:
    wfile.write(df.to_json(orient='records', lines=True))
