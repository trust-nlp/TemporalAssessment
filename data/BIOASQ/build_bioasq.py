import json
import os

# Downloaded all year dataset from BIOASQ chanllenge taskb: from year 2014 to 2024
# note that the data in 2014 challenge dataset 2b is actually the 2013 data, so we map the year from 2013 to 2023
# the training data are renamed and in the format like: BioASQ-trainingDataset2b.json
def process_files(data_folder, output_folder, start_year=2013, end_year=2023):
    id_to_year = {}
    for year in range(start_year, end_year + 1):
        file_name = f"BioASQ-trainingDataset{year - 2011}b.json"
        file_path = os.path.join(data_folder, file_name)

        with open(file_path, 'r') as file:
            data = json.load(file)
            for question in data['questions']:
                question_id = question['id']
                if question_id not in id_to_year:
                    id_to_year[question_id] = year

    # Process the last file and create new JSON
    last_file_path = os.path.join(data_folder, f"BioASQ-trainingDataset{end_year - 2011}b.json")
    with open(last_file_path, 'r') as file:
        data = json.load(file)

    new_file_path = os.path.join(output_folder, "BioASQ.json")
    with open(new_file_path, 'w') as file:
        for question in data['questions']:
            new_question = {
                'id': question['id'],
                'body': question['body'],
                'exact_answer': question.get('exact_answer', None),
                'ideal_answer': question.get('ideal_answer', None),
                'type': question['type'],
                'snippets': ' '.join([snippet['text'] for snippet in question['snippets']]),
                'year': id_to_year[question['id']]
            }
            file.write(json.dumps(new_question) + '\n')
  

data_folder = "/home/weisi/Temporal/data/BIOASQ/BIOASQ_Training"
output_folder = "/home/weisi/TemporalAssessment/data/BIOASQ"
process_files(data_folder, output_folder)