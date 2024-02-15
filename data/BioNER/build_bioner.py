import json
import os
import re

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
                # 如果遇到空行，则视为句子结束
                if not line:
                    # 只有在tokens不为空时才添加到data，避免添加空句子
                    if tokens:
                        data.append({"tokens": tokens, "ner_tags": ner_tags})
                        tokens, ner_tags = [], []  # 重置为下一句做准备
                    continue  # 继续处理下一行

                parts = line.split('\t')
                if len(parts) == 2:
                    token, tag = parts
                    tokens.append(token)
                    ner_tags.append(tag)

            # 处理文件中的最后一个句子（如果文件不是以空行结束）
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

read_and_convert_files(directory,'BioNLP11ID-ggp-IOB', output_directory)
read_and_convert_files(directory,'BioNLP11ID-ggp-IOBES', output_directory)

read_and_convert_files(directory,'BioNLP13GE-IOB', output_directory)
read_and_convert_files(directory,'BioNLP13GE-IOBES', output_directory)