import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import tldextract
from urllib.parse import urlparse, parse_qs
import re
import sys

urls = [
    sys.argv[1]
]

# Create a pandas Series with the URLs and assign index labels
df = pd.Series(urls, index=range(len(urls)))

# Set the name of the Series
df.name = 'url'

# Length
def get_url_length(url):
    return len(url)

# Number of ; _ ? = &
def get_url_punct_count(url):
    return len(re.findall(r'[;_\?=&]', url))

# Digit to Letter ratio
def get_url_dlr(url):
    num_digits = len(re.findall(r'\d', url))
    num_letters = len(re.findall(r'[a-zA-Z]', url))
    if num_letters:
        return num_digits / num_letters
    else:
        return 0

def get_url_feat(X_):
    return pd.concat([
        X_.apply(get_url_length),       # Length
        X_.apply(get_url_punct_count),  # Number of ; _ ? = &
        X_.apply(get_url_dlr)],             # Digit to Letter ratio
    axis=1)

# URL_feat = get_url_feat(X)
# URL_feat.head()

def get_primary_domain(url):
    extracted = tldextract.extract(url)
    return extracted.domain + '.' + extracted.suffix

# Presence of IP
def get_PD_has_IP(url):
    return int(bool(re.match(r'(?:[0-9]{1,3}\.){3}[0-9]{1,3}', url)))

# Length
def get_PD_len(url):
    return len(url)

# Number of digits
def get_PD_digits(url):
    return len(re.findall(r'\d', url))

# Number of non-alphanumeric characters
def get_PD_num_non_alphanumeric(url):
    return len(re.findall(r'[^a-zA-Z0-9]', url))

# Number of hyphens
def get_PD_num_hyphens(url):
    return len(re.findall(r'-', url))

# Number of @s
def get_PD_num_at(url):
    return len(re.findall(r'@', url))

def get_PD_feat(X_):
    PD = X_.apply(lambda x: get_primary_domain(x))
    return pd.concat([
        PD.apply(get_PD_has_IP),               # Presence of IP
        PD.apply(get_PD_len),                  # Length
        PD.apply(get_PD_digits),               # Number of digits
        PD.apply(get_PD_num_non_alphanumeric), # Number of non-alphanumeric characters
        PD.apply(get_PD_num_hyphens),          # Number of hyphens
        PD.apply(get_PD_num_at)],              # Number of @s
        axis=1)

# PD_feat = get_PD_feat(X)
# PD_feat.head()

# Number of dots
def get_SD_num_dots(url):
    return url.count('.')

# Number of subdomains
def get_num_SD(url):
    return sum(1 for subdomain in url.split('.') if subdomain)

def get_SD_feat(X_):
    SD = X_.apply(lambda x: tldextract.extract(x).subdomain)
    return pd.concat([
        SD.apply(get_SD_num_dots),  # Number of dots
        SD.apply(get_num_SD)],      # Number of subdomains
        axis=1)

# SD_feat = get_SD_feat(X)
# SD_feat.head()

# ensure proper parsing of paths
def parse_path(url):
    try:
        if url.startswith('https://') or url.startswith('http://'):
            return urlparse(url)
        else:
            return urlparse('https://' + url)
    except:
        return urlparse(url)

# Number of //
def get_path_num_dbl_fwdslash(url):
    return len(re.findall(r'//', url))

# Number of subdirectories
def get_path_num_subdirs(url):
    return url.count('/')

# Presence of %20
def get_path_has_percent20(url):
    return int('%20' in url)

# Presence of uppercase directories
def get_path_has_uppercase_dirs(url):
    return int(any(any(c.isupper() for c in dir_name) for dir_name in url.split('/')))

# Presence of single character directories
def get_path_has_char_dirs(url):
    return int(any((len(dir_name) == 1) for dir_name in url.split('/')))

# Number of special characters
def get_path_num_special_chars(url):
    return sum(len(re.findall(r'[^A-Za-z0-9]', dir_name)) for dir_name in url.split('/'))

# Number of 0s
def get_path_num_zeroes(url):
    return url.count('0')

# Ratio of uppercase to lowercase characters
def get_path_upper_to_lower_ratio(url):
    upper_count = sum(1 for c in url if c.isupper())
    lower_count = sum(1 for c in url if c.islower())
    if lower_count:
        return upper_count / lower_count
    else:
        return 0

def get_path_feat(X_):
    path_ = X_.apply(lambda x: parse_path(x).path)
    return pd.concat([
        path_.apply(get_path_num_dbl_fwdslash),         # Number of //
        path_.apply(get_path_num_subdirs),              # Number of subdirectories
        path_.apply(get_path_has_percent20),            # Presence of %20
        path_.apply(get_path_has_uppercase_dirs),       # Presence of uppercase directories
        path_.apply(get_path_has_char_dirs),            # Presence of single character directories
        path_.apply(get_path_num_special_chars),        # Number of special characters
        path_.apply(get_path_num_zeroes),               # Number of 0s
        path_.apply(get_path_upper_to_lower_ratio)],    # Ratio of uppercase to lowercase characters
        axis=1)

# path_feat = get_path_feat(X)
# path_feat.head()

# Length
def get_param_length(url):
    return sum(len(value[0]) for value in parse_qs(urlparse(url).query).values())

def get_param_feat(X_):
    return pd.concat([
        X_.apply(get_param_length)],    # Length
        axis=1)

# param_feat = get_param_feat(X)
# param_feat.head()

# Number of queries
def get_num_queries(url):
    return len(parse_qs(urlparse(url).query))

def get_query_feat(X_):
    return pd.concat([
        X_.apply(get_num_queries)
    ])

# query_feat = get_query_feat(X)
# query_feat.head()

def get_lexical_feature_set(X_):
    return pd.concat([
        get_url_feat(X_),
        get_PD_feat(X_),
        get_SD_feat(X_),
        get_path_feat(X_),
        get_param_feat(X_),
        get_query_feat(X_)],
        axis=1)

feature_set = get_lexical_feature_set(df)
print(feature_set)
# Save feature set 
feature_set.to_csv("lexical-features.csv", index=False)