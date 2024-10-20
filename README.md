# Time Matters: Examine Temporal Effects on Biomedical Language Models
The repository of temporal analysis project for the [Time Matters: Examine Temporal Effects on Biomedical Language Models](https://arxiv.org/pdf/2407.17638) in [AMIA 2024 Annual Symposim](https://amia.org/education-events/amia-2024-annual-symposium)

# Table of Contents
 * Data
 * Usage
 * Contact and Citation

# Time-varying Biomedical Data

1. [MIMIC-IV-Notes](https://physionet.org/content/mimic-iv-note/2.2/)
2. BioNLP Shared Task
* [BioNLP09 Shared Task (2009ST)](https://www.nactem.ac.uk/GENIA/SharedTask/)
* [BioNLP11 Epigenetics and Post-translational Modifications (2011EPI)](https://2011.bionlp-st.org)
* [BioNLP11 Infectious Diseases (2011ID) ](https://2011.bionlp-st.org)
* [BioNLP13 Genia Event Extraction (2013GE)](http://www.google.com/url?q=http%3A%2F%2F2013.bionlp-st.org%2F&sa=D&sntz=1&usg=AOvVaw0h0ntV1fsdCdPwDGCqGA06)
3. [BioASQ (task b)](http://participants-area.bioasq.org/datasets/)

## Overview of the data

| Dataset  | Time Intervals                      | Task                  | Labels                         | Data Size         |
|----------|-------------------------------------|-----------------------|--------------------------------|-------------------|
| MIMIC    | 2014 - 2016, 2008 - 2010,2011 - 2013, 2017 -2019           | Phenotype Inference    | Top 50 frequent ICD codes      | 331,794 notes     |
|          |             |                       |                                |                   |
| BioNLP   | 2009ST, 2011EPI, 2011ID,2013GE                     | Information Extraction | IOBES Tags of Protein Entity   | 49,354 entities   |
|          |                     |                       |                                |                   |
| BioASQ   | 2013-2015, 2016-2018,  2019-2020,2021-2023               | Question Answering     | Gold Standard Answer           | 5,046 QA pairs    |
|          |               |                       |                                |                   |

# Usage
1. Data preprocessing and split: 
After downlading data, go to each data folder and run `python build-[data].py` and `python split-[data].py`.
2. Test temporal effect: 
Run `sh [data].sh` to conduct test the performance on all time domain pairs T_{i}-T_{j}
After getting the result, use `python record_[data].py` to get the experiment result summary.
3. Data shift measuring. under the analysis folder, run `/embeddings/[data]_embedings.py' for each [data]
4. Statistic analysis of the performance and the data shift:


note: replace [data] into actual data_path, processing logits for each data is different so need to specify the [data]

# Contact and Citation

<wliu9@memphis.edu>

'''
@misc{liu2024timemattersexaminetemporal,
      title={Time Matters: Examine Temporal Effects on Biomedical Language Models}, 
      author={Weisi Liu and Zhe He and Xiaolei Huang},
      year={2024},
      eprint={2407.17638},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.17638}, 
}
'''