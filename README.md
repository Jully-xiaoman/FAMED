## Overview
This repository contains code necessary to run GAMENetPlus model.

## 🛠️ Requirements
- Pytorch >= 2.9
- Python >= 3.13


## 📁 Data preprocessing
In ./data, you can find the well-preprocessed data in pickle form. Also, it's easy to re-generate the data as follows
1.  download [MIMIC-III data](https://physionet.org/content/mimiciii/1.4/) and put DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv,ADMISSIONS.csv in ./data/mimic-iii/
2.  download [MIMIC-IV data](https://physionet.org/content/mimiciv/3.1/) and put diagnoses_icd.csv, prescriptions.csv, procedures_icd.csv,admissions.csv in ./data/mimic-iv/
3.  run code ./data/mimic-iii/processing_mimic-iii.py to preprocess the MIMIC-III data
4.  run code ./data/mimic-iv/processing_mimic-iii.py to preprocess the MIMIC-IV data

## 🚀 Reproduction
To train and evaluate our model, simply run the shell script `run_mimic_iii.sh`, which includes predefined training commands for MIMIC-III dataset. This script contains the best-performing hyperparameters for MIMIC-III dataset.

Run the following command:
```bash
bash run_mimic_iii.sh
```

To train and evaluate our model, simply run the shell script `run_mimic_iv.sh`, which includes predefined training commands for MIMIC-IV dataset. This script contains the best-performing hyperparameters for MIMIC-IV dataset.

Run the following command:
```bash
bash run_mimic_iv.sh
```
