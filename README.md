# CS3264-Project

### Additional information
- `pip freeze -l > requirements.txt` to update requirements.txt

### Setup
1) Project ran on Python 3.12.2 environment
2) Run `./setup.sh`

### To run features and model
1) Run `./run.sh`

### Feature Engineering

1) HTTP counter
2) HTTPS counter
3) www counter
4) Non-ascii counter
5) Check TLD domain
6) Check authoritative domain
7) Check IP address (if there is)
8) Typosquatting detection via Levenshtein distance on common words from the internet? (might need common authoritative domains)
- Check for some library for this

### Model training

1) Logistic Regression
2) SVM classifier
3) Random Forest classifier