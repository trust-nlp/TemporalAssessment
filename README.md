# TemporalAssessment
Assess Temporal Effects on Machine Learning Models, especially on LLMs

# DataSets

## Classification 

### Amazon Review
Link: https://nijianmo.github.io/amazon/

We downloaded reviews of Health and Personal Care, Digital Music, Video Games.

### MIMIC IV Notes
Link: https://physionet.org/content/mimiciv/2.2/

(Requires training before accessing the dataset, see instructions in the attached link above)
## NER Datasets

### GMB
Link: https://gmb.let.rug.nl/data.php

Download:
```
wget https://gmb.let.rug.nl/releases/gmb-2.2.0.zip
```

### WIESP

Link: https://huggingface.co/datasets/adsabs/WIESP2022-NER

Download:
```python
from datasets import load_dataset
dataset = load_dataset("adsabs/WIESP2022-NER")
```

## QA Datasets
### Amazon QA
Link: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/qa/

We downloaded reviews of Health and Personal Care, Video Games. 

### PubMed QA
Link: https://github.com/pubmedqa/pubmedqa

```bash
wget wget https://github.com/pubmedqa/pubmedqa/blob/master/data/ori_pqal.json
```

```
wget https://github.com/pubmedqa/pubmedqa/blob/master/data/test_ground_truth.json
```
