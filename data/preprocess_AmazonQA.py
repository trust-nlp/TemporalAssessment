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
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip

path='/HDD16TB/weisi/qa_Health_and_Personal_Care.json.gz'
multipath='/HDD16TB/weisi/QA_Health_and_Personal_Care.json.gz'
metapath='/HDD16TB/weisi/meta_Health_and_Personal_Care.json.gz'


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF(path)
metadf=getDF(metapath)
metadf.fillna('x', inplace=True) 
metadata=parse(metapath) #dictionary

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

bidf = df[df['questionType'].isin(['yes/no'])]

tqdm.pandas()#sets up the tqdm progress bar for tracking the progress of the following operation
bidf.question = df.question.progress_apply(lambda x: preprocess(x))
bidf.answer  = df.answer .progress_apply(lambda x: preprocess(x))
#join the list of tokens back into a single string with space-separated words.
df.question = df.question.apply(lambda x: ' '.join(x))
df.answer = df.answer.apply(lambda x: ' '.join(x))

bidf['answerTime'] = pd.to_datetime(bidf['unixTime'], unit='s').dt.strftime('%Y-%m-%d')  #UnixTime has null val
bidf_cleaned = bidf.dropna(subset=['answerTime'])
tqdm.pandas()#sets up the tqdm progress bar for tracking the progress of the following operation
metadf.description= metadf.description.progress_apply(lambda x: preprocess(x))
#metadf.description = metadf.description.apply(lambda x: ' '.join(x))
tqdm.pandas()
metadf.title  = metadf.title .progress_apply(lambda x: preprocess(x))
#metadf.title = metadf.title.apply(lambda x: ' '.join(x))

# Merge the dataframes on the 'asin' column using a left merge
merged_df = pd.merge(bidf_cleaned, metadf[['asin', 'description', 'title', 'salesRank', 'price', 'brand']], on='asin', how='left')

# Export the merged dataframe to a JSON file
merged_df.to_json('combined_data.json', orient='records', lines=True)

combined = pd.read_json('combined_data.json', lines=True)
