# CS3264-Project

Justification for not querying: The user probably does not want to query anything malicious from a website.
Only trusted sources such as DNS Authoritative, TLD, root servers are considered. 

### Additional information
- `pip freeze -l > requirements.txt` to update requirements.txt

### Setup
1) Project ran on Python 3.12.2 environment
2) Run `./setup.sh`

### Model training

1) Random Forest classifier for benign
2) Random Forest classifier for general class classification
3) Random Forest classifier for lexical features
4) Random Forest classifier for lexical and trigrams
5) DistilBERT uncased

### BERT Model 
Download Link: https://drive.google.com/file/d/1yxJGLWCx5lPktDJE9q2VYYwHJW-s6eHH/view?usp=drive_link
Trained using: dataset/Sampled_Data_BERT.csv

### Additional Datasets
#### Training and validation set
https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset

#### Test set
- Phishing test: https://github.com/GregaVrbancic/Phishing-Dataset/blob/master/dataset_full.csv 