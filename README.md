# CS3264-Project

Justification for not querying: The user probably does not want to query anything malicious from a website.
Only trusted sources such as DNS Authoritative, TLD, root servers are considered. 

### Additional information
- `pip freeze -l > requirements.txt` to update requirements.txt

### Setup
1) Project ran on Python 3.12.2 environment
2) Run `./setup.sh`

### To run features and model
1) Run `./run.sh`

### Feature Engineering

1) HTTP counter -> might be a redirection
2) HTTPS counter -> might be a redirection
3) www counter -> might be a redirection
4) Non-ascii counter -> Following RFC 3986 standard of URI, but do not want to discount uncommon symbols (eg %)
5) Length of URL -> longer URL = more likely to be malware
6) Has IP address in URL -> Bypassing DNS address checks, common in malware
7) Domain age -> Registered later, likely to be malware (older domains likely have been caught?), use WHOIS database maintained by ICANN
8) Typosquatting detection via Levenshtein distance on common words from the internet (common authoritative domains from Alexa top 1 million domains)
- Check for some library for this
9) Check google search rank when url is searched

### Model training

1) Logistic Regression
2) SVM classifier
3) Random Forest classifier

### BERT Model 
Download Link: https://drive.google.com/file/d/1yxJGLWCx5lPktDJE9q2VYYwHJW-s6eHH/view?usp=drive_link
Trained using: dataset/Sampled_Data_BERT.csv

### Additional Datasets
#### Training and validation set
https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset

#### Test set
- Phishing test: https://github.com/GregaVrbancic/Phishing-Dataset/blob/master/dataset_full.csv 