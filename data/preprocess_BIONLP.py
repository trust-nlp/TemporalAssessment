import requests
import pandas as pd
import os
from xml.etree import ElementTree
from tqdm import tqdm
import json
#from concurrent.futures import ThreadPoolExecutor

t5path = '/HDD16TB/weisi/all_data/t5'
filenames = [f"questions_{i}.xls" for i in range(3, 7)]


all_pmids = []
for filename in filenames:
    full_path = os.path.join(t5path, filename)
    df = pd.read_excel(full_path)
    all_pmids.extend(df.iloc[:, 3].tolist())

unique_pmids = set(all_pmids)
'''with open("pubmed_ids.txt", "w") as file:
    for pmid in unique_pmids :
        file.write(str(pmid) + "\n")'''

unique_pmidlist = list(unique_pmids) #the set() may give different order of ids; changing to list can be more stable

# E-utilities search year by PubMed ID
base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
api_key = "d83dd5681ba9e58699218fe96723cedc0c08"
id_to_year = {}

#tqdm :record progress ; save data each 100 ids
save_interval = 100
counter = 0
save_pointer = 0 #initialize to 0, if some error occured, change it to saved place 
for id in tqdm(unique_pmidlist, desc="Fetching data"):
    params = {
        "db": "pubmed",
        "id": id,
        "retmode": "xml",
        "api_key": api_key
    }
    response = requests.get(base_url, params=params) #send request ti e-utility efech
    #tree = ElementTree.fromstring(response.content) #may have ParseError: not well-formed (invalid token)

    # parsing xml
    try:
        tree = ElementTree.fromstring(response.content)
    except ElementTree.ParseError:  
        print(f"Error parsing XML for ID {id}. Skipping...")
        continue  # skip error id

    year_elements = tree.findall(".//PubDate/Year")
    if year_elements and year_elements[0].text:
        year = int(year_elements[0].text)
        id_to_year[id] = year #some id may fail to find year(year_elements is empty)
    counter += 1

    if counter % save_interval == 0:
        with open("pmid2year.csv", "a") as f:  # "a"
            for i in range(save_pointer, counter):
                key = unique_pmidlist[i]
                if key in id_to_year:  # make sure this id has found year succesffly(added to dict id_to_year)
                    f.write(f"{key},{id_to_year[key]}\n")
        save_pointer = counter  # update saving point
    
    '''if counter % save_interval == 0:
        with open("partial_results.csv", "w") as f:
            for key, value in id_to_year.items():
                f.write(f"{key},{value}\n")''' # "w" will rewrite the file each interval

#print(id_to_year)
# Save the remaining data
with open("pmid2year.csv", "a") as f:
    for i in range(save_pointer, counter):
        key = unique_pmidlist[i]
        if key in id_to_year:
            f.write(f"{key},{id_to_year[key]}\n")

combined_data = []
for filename in filenames:
    full_path = os.path.join(t5path, filename)
    df = pd.read_excel(full_path)
    combined_data.append(df)

all_data_df = pd.concat(combined_data, ignore_index=True)

# combine tables and output json file
output_data = []
for index, row in all_data_df.iterrows():
    pmid = row[3]
    if pmid in id_to_year:
        single_entry = {
            "question": row[0],   # Assuming the columns are in order
            "long": row[1],
            "short.": row[2],
            "id": pmid,
            "year": id_to_year[pmid]
        }
        output_data.append(single_entry)


with open("bionlp.json", "w") as f:
    for entry in output_data:
        f.write(json.dumps(entry, separators=(',', ':')) + "\n")
  
