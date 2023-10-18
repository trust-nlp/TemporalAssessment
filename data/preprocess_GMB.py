import os
import json
from datetime import datetime
from tqdm import tqdm

BASE_DIR = "/HDD16TB/gmb-2.2.0/data/"

def main():
    result = []
    # folder: from p00 to p99 
    for p_num in range(100):
        p_folder = os.path.join(BASE_DIR, f"p{str(p_num).zfill(2)}") #fill num ij to 00ij

        if os.path.exists(p_folder):
            for d_folder in os.listdir(p_folder):
                met_path = os.path.join(p_folder, d_folder, 'en.met')
                raw_path = os.path.join(p_folder, d_folder, 'en.raw')
                tags_path = os.path.join(p_folder, d_folder, 'en.tags')

                if os.path.exists(met_path) and os.path.exists(raw_path) and os.path.exists(tags_path):
                    with open(met_path, "r", encoding="utf-8") as met_file, open(raw_path, "r", encoding="utf-8") as raw_file, open(tags_path, "r", encoding="utf-8") as tags_file:
                        met_data = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in met_file.readlines()}
                        if "date" not in met_data or not met_data["date"].strip():
                            continue
                        raw_data = raw_file.read().strip()
                        tags_data = [line.split("\t")[:4] for line in tags_file.readlines() if line.strip()]
                        # some met file doesn't has date: for example p00/d0688, so we remove data when it doen't has timestamp
                        date_str = met_data["date"]
                        date_obj = datetime.strptime(date_str, '%B %d, %Y').strftime('%Y-%m-%d')
                        #iso_date = date_obj.isoformat()
                        #print(met_path,date_obj )

                        entry = {
                            "title": met_data["title"],
                            "time": date_obj,
                            "source": met_data["source"],
                            "genre": met_data["genre"],
                            "subcorpus": met_data["subcorpus"],
                            "rawtext": raw_data,
                            "tags": [{"token": tag[0], "pos": tag[1], "lemma": tag[2], "ner": tag[3]} for tag in tags_data]
                        }
                        result.append(entry)
    
    with open('gmb-2.2.0.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
