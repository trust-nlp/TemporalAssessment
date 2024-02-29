import json
import os
import re
import pandas as pd

'''def is_sentence_end(prev_token, token, next_token):
    if token in {'!', '?'}:
        return True
    if token == '.':
        # if there are no next token，or next token is capitalized
        if not next_token or re.match(r'^[A-Z]', next_token):
            if not re.match(r'^[A-Z]$', prev_token): # For abbreviation like name M.M.Lightfoote
                return True  
        else:
            return False 
    return False  '''

def read_and_convert_files(directory, filename, output_directory):
    splits = ['devel.tsv', 'train.tsv', 'test.tsv']
    data = []

    for split in splits:
        filepath = os.path.join(directory, filename, split)
        with open(filepath, 'r', encoding='utf-8') as file:
            tokens, ner_tags = [], []
            for line in file:
                line = line.strip()
                # a blank line is the end of last sentence
                if not line:
                    # avoid adding empty token:
                    if tokens:
                        data.append({"tokens": tokens, "ner_tags": ner_tags})
                        tokens, ner_tags = [], []  
                    continue  # next line

                parts = line.split('\t')
                if len(parts) == 2:
                    token, tag = parts
                    tokens.append(token)
                    ner_tags.append(tag)

            # last sentence
            if tokens:
                data.append({"tokens": tokens, "ner_tags": ner_tags})

    output_file_path = os.path.join(output_directory, f'{filename}.json')
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for entry in data:
            outfile.write(json.dumps(entry) + '\n')


    # Output file path
    output_file_path = os.path.join(output_directory, f'{filename}.json')
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for entry in data:
            outfile.write(json.dumps(entry) + '\n')




directory = '/HDD16TB/weisi/MTL-Bioinformatics-2016-master/data/'
#filename='BioNLP09-IOB'
#filename2='BioNLP09-IOBES'
output_directory = '/home/weisi/TemporalAssessment/data/BioNER/Protein'


os.makedirs(output_directory, exist_ok=True)

read_and_convert_files(directory,'BioNLP09-IOB', output_directory)
read_and_convert_files(directory,'BioNLP09-IOBES', output_directory)

read_and_convert_files(directory,'BioNLP11EPI-IOB', output_directory)
read_and_convert_files(directory,'BioNLP11EPI-IOBES', output_directory)

read_and_convert_files(directory,'BioNLP11ID-IOB', output_directory)
read_and_convert_files(directory,'BioNLP11ID-IOBES', output_directory)

read_and_convert_files(directory,'BioNLP13GE-IOB', output_directory)
read_and_convert_files(directory,'BioNLP13GE-IOBES', output_directory)


basepath = '/home/weisi/TemporalAssessment/data/BioNER/Protein'
prefixes = [
    'BioNLP09-IOBES',
    'BioNLP11EPI-IOBES',
    'BioNLP11ID-IOBES',
    'BioNLP13GE-IOBES'
]

dataframes = [] 

for prefix in prefixes:
    domain = prefix.split('-')[0]  
    filepath = os.path.join(basepath, f"{prefix}.json") 
    print(prefix)
    data=[]
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line))
        #data = json.load(file)
    
    
    for item in data:
        
        item['text'] = ' '.join(item['tokens'])  # 将 tokens TO string
        item['domain'] = domain  # add domain 
    
    df = pd.DataFrame(data)
    dataframes.append(df)

merged_df = pd.concat(dataframes, ignore_index=True)

output_filepath = os.path.join(basepath, "BIONER-IOBES.json")
merged_df.to_json(output_filepath, orient='records', lines=True)
