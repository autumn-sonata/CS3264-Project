# CS3264-Project

### Package Installation
- `pip freeze -l > requirements.txt` to update requirements.txt

### Setup
1) Project ran on Python 3.12.2 environment
2) Run `./setup.sh`

### Model training

1) Random Forest classifier for benign (`rf-minimal`)
2) Random Forest classifier for general class classification
3) Random Forest classifier for lexical features
4) Random Forest classifier for lexical and trigrams
5) DistilBERT Cased

### Models
Models for `rf-general`, `rf-lexical`, `rf-minimal` and `distilBERT` cased can be found in:
https://drive.google.com/drive/u/0/folders/1v5IFIY7J0EJPg6VVZXGeGp1DY9VGPuvp

### Additional Datasets
#### Training and validation set
https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset

#### Test set
- Phishing test: https://phishtank.org/developer_info.php
- Benign test: https://www.kaggle.com/datasets/siddharthkumar25/malicious-and-benign-urls

