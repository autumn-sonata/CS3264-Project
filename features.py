import pandas as pd
import re

df = pd.read_csv("datasets/malicious_phish.csv")
X, y = pd.DataFrame(), pd.DataFrame()

def classification_type(type):
    '''
    Convert classification type into values:
        1) Benign = 0
        2) Defacement = 1
        3) Phishing = 2
        4) Malware = 3    
    '''
    if type == "benign":
        return 0
    elif type == "defacement":
        return 1
    elif type == "phishing":
        return 2
    elif type == "malware":
        return 3
    else:
        print(f"Unable to find proper type: {type}")

y['type_val'] = df['type'].apply(classification_type)

# Feature engineering

# 1) HTTP count
def get_http(url):
    return len(re.findall(r"http://", url))

X['http'] = df['url'].apply(get_http)

# 2) HTTPS count
def get_https(url):
    return len(re.findall(r"https://", url))

X['https'] = df['url'].apply(get_https)

# 3) www count
def get_www(url):
    return len(re.findall(r"www", url))

X['www'] = df['url'].apply(get_www)

# 4) Non-ASCII counter
def get_num_non_ascii(url):
    count = 0
    for ch in url:
        if ord(ch) > 127:
            count += 1
    return count

X['non_ascii'] = df['url'].apply(get_num_non_ascii)


# Maybe do evaluation on features? Eg chi2 test, select k best 

# Store dataframe into CSV file for modelling
X.to_csv("datasets/feature_updated_dataset_X.csv", index=False)
y.to_csv("datasets/feature_updated_dataset_y.csv", index=False)
