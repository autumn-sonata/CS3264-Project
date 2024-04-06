import math
import pandas as pd
import re
import whois
from datetime import datetime
import time
from googlesearch import search
import tldextract
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt 
import sys

df = pd.read_csv("datasets/malicious_phish.csv")

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

y = df['type'].apply(classification_type)

## Analysis of counts
def count_analysis(X, y, feature_name):
    unique_y = np.unique(y)
    fig, axs = plt.subplots(len(unique_y), 1, figsize=(8, len(unique_y) * 4))
    for i in unique_y:
        indices = np.where(y == i)[0]
        counts_i = X[indices]
        unique_vals, counts = np.unique(counts_i, return_counts=True)
        axs[i].bar(unique_vals, counts, width=0.5)
        # axs[i].set_xticks(np.arange(min(unique_vals), max(unique_vals)+1, 1))
        axs[i].set_xlabel(f"Number of {feature_name} in URL: {i}")
        axs[i].set_ylabel(f"Number of training examples: {i}")
        axs[i].set_title(f"Number of training examples by number of {feature_name} in URL: {i}")

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(f"plots/{feature_name}_distribution.png")

# Feature engineering

# 1) HTTP count
def get_http(url):
    return len(re.findall(r"http://", url))

## Analysis of HTTP count
http_counts = df['url'].apply(get_http).values
csr_http = csr_matrix(http_counts).T
#count_analysis(http_counts, y, "HTTP")


# 2) HTTPS count
def get_https(url):
    return len(re.findall(r"https://", url))

## Analysis of HTTPS count
https_counts = df['url'].apply(get_https).values
csr_https = csr_matrix(https_counts).T
#count_analysis(https_counts, y, "HTTPS")

# 3) www count
def get_www(url):
    return len(re.findall(r"www", url))

## Analysis of www count
www_counts = df['url'].apply(get_www).values
csr_www = csr_matrix(www_counts).T
#count_analysis(www_counts, y, "www")

# 4) Non-ASCII counter NOT USEFUL
# def get_num_non_ascii(url):
#     count = 0
#     for ch in url:
#         if ord(ch) > 127:
#             count += 1
#     return count

# non_ascii_counts = df['url'].apply(get_num_non_ascii).values
# csr_non_ascii = csr_matrix(non_ascii_counts).T
# count_analysis(non_ascii_counts, y, "non_ascii")

# 5) URL length
def get_url_length(url):
    return len(url)

url_length_count = df['url'].apply(get_url_length)
csr_url_len = csr_matrix(url_length_count).T
#count_analysis(url_length_count, y, "url_length")

# 6) IP address presence: NOT USEFUL
# def has_ip_address(url):
#     ip_address_re = r"(?:\d{1,3}\.){3}\d{1,3}"
#     return bool(re.match(ip_address_re, url))

# ip_addr_presence = df['url'].apply(has_ip_address).values
# csr_has_ip = csr_matrix(ip_addr_presence).T
# count_analysis(ip_addr_presence, y, "ip address")

# 7) Google search ranking
# def get_google_search_rank(url):
#     num_results = 5
#     searches = search(url, num_results=num_results)
#     time.sleep(1) # rate limiting to mitigate HTTP 429 Too many requests
#     for i, res in enumerate(searches):
#         if tldextract.extract(res) == tldextract.extract(url):
#             return i

#     return num_results

# csr_gsearch_rank = csr_matrix(df['url'].apply(get_google_search_rank).values).T

# 8) Domain age
# def get_domain_age(url):
    # data = whois.whois('google.com') # might have rate limiting

# csr_domain_age = df['url'].apply(get_domain_age) # not completed

# 9) Get minimum Levenshtein distance from top 1 million websites?

# 10) Domain and subdomain and suffix encoding: KEEPS GOING OUT OF MEMORY
#def domain_subdomain_suffix_encoding(df):
    #extract_res = df.apply(lambda url: tldextract.extract(url))

    #domain = extract_res.map(lambda obj: obj.domain)
    #subdomain = extract_res.map(lambda obj: obj.subdomain)
    #suffix = extract_res.map(lambda obj: obj.suffix)

    # print(domain.nunique(), subdomain.nunique(), suffix.nunique())

    #domain_vec, subdomain_vec, suffix_vec = CountVectorizer(binary=True), \
        #CountVectorizer(binary=True), CountVectorizer(binary=True)

    #domain = domain_vec.fit_transform(domain)
    #subdomain = subdomain_vec.fit_transform(subdomain)
    #suffix = suffix_vec.fit_transform(suffix)

    #return domain#, subdomain, suffix

# 11) Number of digits
def get_num_digits(url):
    return len(re.findall(r'\d', url))

digit_count = df['url'].apply(get_num_digits)
csr_digit_count = csr_matrix(digit_count).T
#count_analysis(digit_count, y, "digit")

# 12) Number of %
def get_num_percent(url):
    return len(re.findall(r'%', url))

percentage_count = df['url'].apply(get_num_percent)
csr_percentage_count = csr_matrix(percentage_count).T
#count_analysis(percentage_count, y, "%")

# 13) Number of .
def get_num_dot(url):
    return len(re.findall(r'.', url))
dot_count = df['url'].apply(get_num_dot)
csr_dot_count = csr_matrix(dot_count).T
#count_analysis(dot_count, y, ".")

# 14) Number of /
def get_num_backslash(url):
    return len(re.findall(r'/', url))
bs_count = df['url'].apply(get_num_backslash)
csr_bs_count = csr_matrix(bs_count).T
#count_analysis(bs_count, y, "/")

# 15) Number of -
def get_num_dash(url):
    return len(re.findall(r'-', url))
dash_count = df['url'].apply(get_num_dash)
csr_dash_count = csr_matrix(dash_count).T
#count_analysis(dash_count, y, "-")

# 16) URL entropy
def get_url_entropy(url):
    url_str = url.strip()
    prob = [float(url_str.count(c)) / len(url_str) for c in dict.fromkeys(list(url_str))]
    entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
    return entropy
url_entropy = df['url'].apply(get_url_entropy)
csr_url_entropy = csr_matrix(url_entropy).T
#count_analysis(url_entropy, y, "entropy")

# 17) Number of parameters
def get_num_params(url):
    params = url.split('&')
    return len(params) - 1
url_num_params = df['url'].apply(get_num_params)
csr_url_num_params = csr_matrix(url_num_params).T

# 18) Number of subdomains
def get_num_subdomains(url):
  if 'http' in url:
    subdomains = url.split('http')[-1].split('//')[-1].split('/')
  else:
    subdomains = url.split('https')[-1].split('//')[-1].split('/')
  return len(subdomains)-1
url_num_subdomains = df['url'].apply(get_num_subdomains)
csr_url_num_subdomains = csr_matrix(url_num_subdomains).T

# 19) Domain extension
def get_domain_extension(url):
    ext = url.split('.')[-1].split('/')[0]
    return ext
domain_extension = df['url'].apply(get_domain_extension)
domain_extension = domain_extension.factorize()[0]
csr_domain_extension = csr_matrix(domain_extension).T

X = hstack([csr_http, csr_https, csr_www, csr_url_len, csr_digit_count, 
      csr_percentage_count, csr_dot_count, csr_bs_count, csr_dash_count, 
      csr_url_entropy, csr_url_num_params, csr_url_num_subdomains, csr_domain_extension])

# Adding headers so it's easier to read when doing feature analysis 
column_names = ['http', 'https', 'www', 'url_length', 'digit_count', 'percentage_count', 'dot_count', 'bs_count', 'dash_count', 'url_entropy', 'params_count', 'subdomains_count', 'domain_extension']
dfX = pd.DataFrame(X.toarray(), columns=column_names)
dfy = pd.DataFrame({'type_val': y})


# Maybe do evaluation on features? Eg chi2 test, select k best, PCA 
# Store dataframe into CSV file for modelling
dfX.to_csv("datasets/feature_updated_dataset_X.csv", index=False)
dfy.to_csv("datasets/feature_updated_dataset_y.csv", index=False)