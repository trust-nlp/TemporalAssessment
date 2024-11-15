# Time Matters: Examine Temporal Effects on Biomedical Language Models
This repository accompanies the project: [Time Matters: Examine Temporal Effects on Biomedical Language Models](https://arxiv.org/pdf/2407.17638) presented in [AMIA 2024 Annual Symposim](https://amia.org/education-events/amia-2024-annual-symposium)
It explores how biomedical language models' performance degrades over time due to shifts in data between training and deployment. The project covers three biomedical tasks, using performance metrics, data drift measurements, and statistical analyses to quantify the temporal effects.
Results show that time impacts model performance, with varying degrees of degradation across tasks.

<img width="1136" alt="time_matters" src="https://github.com/user-attachments/assets/a8fd501a-bdba-4dcd-b7c8-4110790ddf59">


# Table of Contents
 * Environment Setup
 * Time-varying Biomedical Data
 * Usage
 * Contact and Citation

# Environment Setup
1. Platform:

* Ubuntu 22.04
* Anaconda, Python 3.10.13
* Linux Kernel: 6.8.0-40-generic

2. Run the following commands to create the environment:

* `conda env create -f environment.yml`
* `conda activate tempo0`

# Time-varying Biomedical Data

1. [MIMIC-IV-Notes](https://physionet.org/content/mimic-iv-note/2.2/)
2. BioNLP Shared Task (BioNER)
      * [BioNLP09 Shared Task (2009ST)](https://www.nactem.ac.uk/GENIA/SharedTask/)
      * [BioNLP11 Epigenetics and Post-translational Modifications (2011EPI)](https://2011.bionlp-st.org)
      * [BioNLP11 Infectious Diseases (2011ID) ](https://2011.bionlp-st.org)
      * [BioNLP13 Genia Event Extraction (2013GE)](http://www.google.com/url?q=http%3A%2F%2F2013.bionlp-st.org%2F&sa=D&sntz=1&usg=AOvVaw0h0ntV1fsdCdPwDGCqGA06)
3. [BioASQ (task b)](http://participants-area.bioasq.org/datasets/)

## Overview of the data

| Dataset  | Time Intervals                      | Task                  | Labels                         | Data Size         |
|----------|-------------------------------------|-----------------------|--------------------------------|-------------------|
| MIMIC    | 2014 - 2016, 2008 - 2010, 2011 - 2013, 2017 -2019 | Phenotype Inference    | Top 50 frequent ICD codes      | 331,794 notes     |
| BioNER   | 2009ST, 2011EPI, 2011ID, 2013GE| Information Extraction | IOBES Tags of Protein Entity   | 49,354 entities   |
| BioASQ   | 2013-2015, 2016-2018, 2019-2020, 2021-2023| Question Answering     | Gold Standard Answer           | 5,046 QA pairs    |


# Usage
1. Data preprocessing and split: 
After downlading data, go to each data folder and run `python build-[data].py` and `python split-[data].py`.
2. Test temporal effect: 
Run `sh [data].sh` to conduct test the performance on all time domain pairs.
After getting the result, use `python record_[data].py` to get the experiment result summary.
3. Data shift measuring:
* Run `analysis/[encoder]_embedings.py` to get the text embedding for semantic shift evaluation. Here, [encoder] can be selected from general domain encoders like SBERT and USE, or biomedical domain encoders like BioLORD and MedCPT.
* Run `analysis/token_level_metrics.py` to measure the token-level data shift, using metrics such as TF-IDF cosine similarity and Jaccard similarity.
4. Statistic analysis of the performance and the data shift.
Run `analysis/statistic_analysis.py` to explore the relationships between language model performance and data shifts.

Note: Replace [data] with the actual data path. Since processing differs for each dataset, ensure to specify the correct [data] and file path.

# Contact and Citation

For any inquiries, feel free to contact the author at: <wliu9@memphis.edu>

To cite this work:
```
@misc{liu2024timemattersexaminetemporal,
      title={Time Matters: Examine Temporal Effects on Biomedical Language Models}, 
      author={Weisi Liu and Zhe He and Xiaolei Huang},
      year={2024},
      eprint={2407.17638},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.17638}, 
}
```
