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

from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

hosp_dir = '/HDD16TB/Datasets/physionet.org/files/mimiciv/2.2/hosp/'
note_path = '/HDD16TB/Datasets/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv'
admission_path = hosp_dir + 'admissions.csv'
patients_path=hosp_dir + 'patients.csv'

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
    doc = re.sub(r"(\b)(d|[dD])\.?(r|[rR])\.?(\b)", " ", doc)  # remove Dr abbreviation (I guess it's because the name of Dr is hidded so we just drop Dr. as well)
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

## notes table
notes = pd.read_csv(
    note_path, dtype=str, usecols=['note_id', 'subject_id', 'hadm_id', 'charttime', 'storetime', 'text'])
notes.charttime = pd.to_datetime(notes.charttime)
notes.subject_id = notes.subject_id.apply(lambda x: x.strip())
notes.fillna('x', inplace=True)

tqdm.pandas()#sets up the tqdm progress bar for tracking the progress of the following operation
notes.text = notes.text.progress_apply(lambda x: preprocess(x))  # track progress
# filter out small documents
notes = notes[notes.text.apply(len) >= 30]#rows with text shorter than 30 characters are removed
notes.text = notes.text.apply(lambda x: ' '.join(x))#join the list of tokens back into a single string with space-separated words.


## patient table
patient_set = set(notes.subject_id)
patients = pd.read_csv(patients_path, dtype=str)
patients.subject_id = patients.subject_id.apply(lambda x: x.strip())
patients = patients[patients.subject_id.isin(patient_set)]
patients.fillna('x', inplace=True)

# convert to a dictionary: 
patients_dict = dict((z[0], list(z[1:])) for z in zip(
    patients.subject_id, patients.gender, patients['anchor_age'], 
    patients['anchor_year_group'], patients.dod, patients['anchor_year']
))

# add mortality label, in-hospital mortality, out-of-hospital mortality, mortality
for uid in patients_dict:
    patients_dict[uid].extend([0,0,0])
    if patients_dict[uid][3] != 'x':
        patients_dict[uid][-1] = 1
        death_year = int(patients_dict[uid][3].split('-')[0])
        if death_year > int(patients_dict[uid][4]):
            patients_dict[uid][-2] = 1  # out-hospital mortality
        else:
            patients_dict[uid][-3] = 1  # in-hospital mortality
    else:
        continue

## admission table
admits = pd.read_csv(admission_path, dtype=str)
admits.race = admits.race.fillna('OTHER')  # replace N/A values
admits.fillna('x', inplace=True) #fills missing (NaN) values in the with the character 'x'
admits_dict = dict((z[0], list(z[1:])) for z in zip(
    admits.subject_id, admits.race, admits['marital_status'], admits.insurance))

diagnoses=pd.read_csv(hosp_dir + 'diagnoses_icd.csv', dtype=str)

# record codes
dfcodes = dict()
icd9_count = []
icd10_count = []
hadm_set = set(notes.hadm_id)
with open(hosp_dir + 'diagnoses_icd.csv') as dfile:
    cols = [col.replace('"', '').strip() for col in dfile.readline().strip().split(',')]
    #cols = dfile.readline().strip() is string :"subject_id,hadm_id,seq_num,icd_code,icd_version" 
    #neefd to replace " and split using commas 
    subj_idx = cols.index('subject_id')
    hadm_idx = cols.index('hadm_id')
    icd_idx = cols.index('icd_code')
    version_idx = cols.index('icd_version')
    
    
    for line in dfile:
        line = [item.replace('"', '').strip() for item in line.strip().split(',')]
        if len(line) != len(cols):
            continue  # skip this line if it has missing value
        if line[subj_idx] not in patient_set:
            continue # skip if not in notes.subject_id
        if line[hadm_idx] not in hadm_set:
            continue # skip if not in hadm_id list

        code_id = '{0}-{1}'.format(line[subj_idx], line[hadm_idx])
        # code_id = line[hadm_idx].strip()
        if code_id not in dfcodes:
            dfcodes[code_id] = {'icd9':[], 'icd10':[]}
        if int(line[version_idx]) == 9:
            dfcodes[code_id]['icd9'].append(line[icd_idx])
            icd9_count.append(line[icd_idx])
        else:
            dfcodes[code_id]['icd10'].append(line[icd_idx])
            icd10_count.append(line[icd_idx])

# loop through each note
results = dict()
total_entries = 0
for index, row in tqdm(notes.iterrows(), total=len(notes)):
    # uid = row['SUBJECT_ID'] + '-' + row['HADM_ID']
    uid = row['subject_id'].strip()
    hadm_id = row['hadm_id'].strip()
    code_id = '{0}-{1}'.format(uid, hadm_id)
    if code_id not in dfcodes:
        continue

    if uid not in results:
        u_age = int(patients_dict[uid][1])
        results[uid] = {
            'uid': uid,
            # for time effect analysis
            'time': patients_dict[uid][2],
            # calculate from the current stay and patient's DOB
            'age': u_age,
            'gender': patients_dict[uid][0],  # first value is the gender
            'ethnicity': admits_dict[uid][0],  # ethnicity,
            'maritial_status': admits_dict[uid][1],
            'insurance': admits_dict[uid][2],
            'mortality': patients_dict[uid][-1],
            'in-mortality': patients_dict[uid][-3],
            'out-mortality': patients_dict[uid][-2],
            'docs': list(),  # collect all patient notes
        }

    results[uid]['docs'].append({
        'doc_id': row['note_id'],
        'date': row['charttime'].strftime('%Y-%m-%d'),
        'hadm_id': hadm_id,
        'text': row['text'],
        'tags': dfcodes[code_id],
    })
    total_entries += 1

with open('mimic.json', 'w') as wfile:
    for uid in tqdm(results, total=len(results)):
        for doc in results[uid]['docs']:
            entity = {
                'uid': uid, 'did': doc['doc_id'], 'text': doc['text'], 'time': results[uid]['time'],
                'gender': results[uid]['gender'], 'age': results[uid]['age'], 'insurance': results[uid]['insurance'],
                'ethnicity': results[uid]['ethnicity'], 'maritial_status': results[uid]['maritial_status'],
                'doc-date': doc['date'], 'mortality': results[uid]['mortality'], 
                'in-mortality': results[uid]['in-mortality'], 'out-mortality': results[uid]['out-mortality'],
                'icd9': doc['tags']['icd9'], 'icd10': doc['tags']['icd10'], 'hadm_id': doc['hadm_id']
            }
            if len(entity) == 0:
                continue
            wfile.write(json.dumps(entity) + '\n')  # use the mortality as the label to simplify the experiments
