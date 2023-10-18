from datasets import load_dataset
import json

dataset = load_dataset("adsabs/WIESP2022-NER")

splits = ["train", "test", "validation"]

for split in splits:
    if split in dataset:
        filename = f"WIESP2022-NER-{split}.jsonl"
        with open(filename, "w", encoding="utf-8") as file:
            for item in dataset[split]:
                year = item["bibcode"][:4]
                item["time"] = year  #add key "time" by extacting year from bibcode.
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        print(f"'{split}' subset not found in the dataset.")  
