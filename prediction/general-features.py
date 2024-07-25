
import math
from urllib.parse import urlparse, parse_qs
import pandas as pd
import re
# import whois
from datetime import datetime
import time
# from googlesearch import search
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

def load_top_1m_set(csv_file):
    """
    Load the top 1 million URLs from a CSV file into a set.
    """
    top_1m_set = set()
    with open(csv_file, 'r') as file:
        for line in file:
            url = line.strip()  # Remove leading/trailing whitespace
            top_1m_set.add(url)
    return top_1m_set

top1m = load_top_1m_set("D:\projects\cs3264\datasets\\top-1m.csv")

url = pd.Series([sys.argv[1]])
# Feature engineering
# ==== Whole URL Features ==== #
# 3) www count
def get_www(url):
    return len(re.findall(r"www", url))

## Analysis of www count
www_counts = url.apply(get_www).values
csr_www = csr_matrix(www_counts).T
#count_analysis(www_counts, y, "www")

# 5) URL length
def get_url_length(url):
    return len(url)

url_length_count = url.apply(get_url_length)
csr_url_len = csr_matrix(url_length_count).T
#count_analysis(url_length_count, y, "url_length")

# 6) IP address presence: NOT USEFUL
# def has_ip_address(url):
#     ip_address_re = r"(?:\d{1,3}\.){3}\d{1,3}"
#     return bool(re.match(ip_address_re, url))

# ip_addr_presence = url.apply(has_ip_address).values
# csr_has_ip = csr_matrix(ip_addr_presence).T
# count_analysis(ip_addr_presence, y, "ip address")

# 11) Number of digits
def get_num_digits(url):
  return len(re.findall(r'\d', url))

digit_count = url.apply(get_num_digits)
csr_digit_count = csr_matrix(digit_count).T
#count_analysis(digit_count, y, "digit")

# 12) Number of %
def get_num_percent(url):
  return len(re.findall(r'%', url))

percentage_count = url.apply(get_num_percent)
csr_percentage_count = csr_matrix(percentage_count).T
#count_analysis(percentage_count, y, "%")

# 13) Number of .
def get_num_dot(url):
  return len(re.findall(r'.', url))
dot_count = url.apply(get_num_dot)
csr_dot_count = csr_matrix(dot_count).T
#count_analysis(dot_count, y, ".")

# 14) Number of /
def get_num_backslash(url):
  return len(re.findall(r'/', url))
bs_count = url.apply(get_num_backslash)
csr_bs_count = csr_matrix(bs_count).T
#count_analysis(bs_count, y, "/")

# 15) Number of -
def get_num_dash(url):
  return len(re.findall(r'-', url))
dash_count = url.apply(get_num_dash)
csr_dash_count = csr_matrix(dash_count).T
#count_analysis(dash_count, y, "-")

# 16) URL entropy
def get_url_entropy(url):
  url_str = url.strip()
  prob = [float(url_str.count(c)) / len(url_str) for c in dict.fromkeys(list(url_str))]
  entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
  return entropy
url_entropy = url.apply(get_url_entropy)
csr_url_entropy = csr_matrix(url_entropy).T
#count_analysis(url_entropy, y, "entropy")

# 17) Number of parameters
def get_num_params(url):
  params = url.split('&')
  return len(params) - 1
url_num_params = url.apply(get_num_params)
csr_url_num_params = csr_matrix(url_num_params).T

# 18) Number of subdomains
def get_num_subdomains(url):
  extracted = tldextract.extract(url)
  return len(extracted.subdomain.split('.'))
url_num_subdomains = url.apply(get_num_subdomains)
csr_url_num_subdomains = csr_matrix(url_num_subdomains).T

# 19) Domain extension
def get_domain_extension(url):
  extracted = tldextract.extract(url)
  if extracted.suffix == '':
    return 'None'
  return extracted.suffix

domain_extension = url.apply(get_domain_extension)
domain_extension = domain_extension.factorize()[0]
csr_domain_extension = csr_matrix(domain_extension).T
# keep track of domain to index mapping
domain_index_df = pd.DataFrame({'Domain': url.apply(get_domain_extension), 'Index': domain_extension})

# 20) Number of semicolons
def get_num_semicolon(url):
  return len(re.findall(r';', url))
semicolon_count = url.apply(get_num_semicolon)
csr_semicolon_count = csr_matrix(semicolon_count).T

# 21) Number of underscores
def get_num_underscores(url):
  return len(re.findall(r'_', url))
underscores_count = url.apply(get_num_underscores)
csr_underscores_count = csr_matrix(underscores_count).T

# 22) Number of question marks
def get_num_questionmarks(url):
  return len(re.findall(r'\?', url))
questionmarks_count = url.apply(get_num_questionmarks)
csr_questionmarks_count = csr_matrix(questionmarks_count).T

# 23) Number of equals
def get_num_equals(url):
  return len(re.findall(r'=', url))
equals_count = url.apply(get_num_equals)
csr_equals_count = csr_matrix(equals_count).T

# 24) Number of ampersands
def get_num_ampersands(url):
  return len(re.findall(r'\&', url))
ampersands_count = url.apply(get_num_ampersands)
csr_ampersands_count = csr_matrix(ampersands_count).T

# 25) Digit to letter ratio
def get_digit_letter_ratio(url):
  num_digits = len(re.findall(r'\d', url))
  num_letters = len(re.findall(r'[a-zA-Z]', url))
  if num_letters == 0:
      return 0
  return num_digits / num_letters
digit_letter_ratio = url.apply(get_digit_letter_ratio)
csr_digit_letter_ratio = csr_matrix(digit_letter_ratio).T

# ===== Primary Domain Features ===== #

# 26) Primary Domain: Number of non-alphanumeric characters
def pd_get_num_count(url):
  extracted = tldextract.extract(url)
  primary_domain = extracted.domain + '.' + extracted.suffix
  return len(re.findall(r'\d', primary_domain))
pd_num_count = url.apply(pd_get_num_count)
csr_pd_num_count = csr_matrix(pd_num_count).T

# 27) Primary Domain: Number of non-alphanumeric characters
def pd_get_num_non_alphanumeric(url):
  extracted = tldextract.extract(url)
  primary_domain = extracted.domain + '.' + extracted.suffix
  return len(re.findall(r'[^a-zA-Z0-9]', primary_domain))
pd_non_alphanumeric_count = url.apply(pd_get_num_non_alphanumeric)
csr_pd_non_alphanumeric_count = csr_matrix(pd_non_alphanumeric_count).T

# 28) Primary Domain: Number of @
def pd_get_num_at(url):
  extracted = tldextract.extract(url)
  primary_domain = extracted.domain + '.' + extracted.suffix
  return len(re.findall(r'@', primary_domain))
pd_at_count = url.apply(pd_get_num_at)
csr_pd_at_count = csr_matrix(pd_at_count).T

# 29) Primary Domain: Number of hyphens
def pd_get_hyphen_count(url):
  extracted = tldextract.extract(url)
  primary_domain = extracted.domain + '.' + extracted.suffix
  return len(re.findall(r'-', primary_domain))
pd_hyphen_count = url.apply(pd_get_hyphen_count)
csr_pd_hyphen_count = csr_matrix(pd_hyphen_count).T

# 30) Primary Domain: In top alexa 1m
def pd_get_in_alexa_top_1m(url):
  extracted = tldextract.extract(url)
  primary_domain = extracted.domain + '.' + extracted.suffix
  return primary_domain in top1m
pd_in_alex_top_1m = url.apply(pd_get_in_alexa_top_1m)
csr_pd_in_alex_top_1m = csr_matrix(pd_in_alex_top_1m).T

# ===== Path Features ===== #
# 31) Number of //
def get_path_num_double_slash(url):
  path = urlparse(url).path
  return len(re.findall(r'//', path))
path_double_slash_count = url.apply(get_path_num_double_slash)
csr_path_double_slash_count = csr_matrix(path_double_slash_count).T

# 32) presence of %20
def has_percent20(url):
  return '%20' in url
percent20_presence = url.apply(has_percent20)
csr_percent20_presence = csr_matrix(percent20_presence).T

# 33) presence of uppercase directories
def uppercase_dirs_count(url):
  path = urlparse(url).path
  return sum(1 for dir_name in path.split('/') if any(c.isupper() for c in dir_name))
uppercase_dirs = url.apply(uppercase_dirs_count)
csr_uppercase_dirs = csr_matrix(uppercase_dirs).T

# 34) presence of single character directories
def single_char_dirs_count(url):
  path = urlparse(url).path
  return sum(1 for dir_name in path.split('/') if len(dir_name) == 1)
single_char_dirs = url.apply(single_char_dirs_count)
csr_single_char_dirs = csr_matrix(single_char_dirs).T

# 35) presence of special characters in path
def special_chars_count(url):
  path = urlparse(url).path
  return sum(1 for c in path if not c.isalnum() and c != '/')
path_count_special_chars = url.apply(special_chars_count)
csr_path_count_special_chars = csr_matrix(path_count_special_chars).T

# 36) count of zeroes in path
def zeroes_count(url):
  path = urlparse(url).path
  return path.count('0')
path_zeroes_count = url.apply(zeroes_count)
csr_path_zeroes_count = csr_matrix(path_zeroes_count).T

# 37) presence of uppercase to lowercase ratio in path
def uppercase_to_lowercase_ratio(url):
  # Extract the path from the URL
  path = urlparse(url).path
  # Calculate the ratio of uppercase to lowercase characters
  uppercase_count = sum(1 for c in path if c.isupper())
  lowercase_count = sum(1 for c in path if c.islower())
  return uppercase_count / lowercase_count if lowercase_count > 0 else 0
path_uppercase_to_lowercase_ratio = url.apply(uppercase_to_lowercase_ratio)
csr_path_uppercase_to_lowercase_ratio = csr_matrix(path_uppercase_to_lowercase_ratio).T

# 38) get length of params
def params_get_length(url):
  query = urlparse(url).query
  return len(query)
params_length = url.apply(params_get_length)
csr_params_length = csr_matrix(params_length).T

# 39) get number of queries
def queries_get_count(url):
  query = urlparse(url).query
  return len(parse_qs(query))
queries_count = url.apply(queries_get_count)
csr_queries_count = csr_matrix(queries_count).T

# Reduced: feature_updated_dataset_X_reduced
X = hstack([csr_www, csr_url_len, csr_digit_count, 
      csr_dot_count, csr_bs_count, csr_dash_count, 
      csr_url_entropy, csr_url_num_params, csr_url_num_subdomains, csr_domain_extension,
      csr_underscores_count, csr_questionmarks_count,
      csr_equals_count, csr_ampersands_count, csr_digit_letter_ratio, 
      
      csr_pd_num_count, csr_pd_non_alphanumeric_count,
      
      csr_uppercase_dirs, csr_path_count_special_chars,
      csr_path_uppercase_to_lowercase_ratio, csr_params_length,
      csr_queries_count])


# Adding headers so it's easier to read when doing feature analysis 
column_names = ['www', 'url_length', 'digit_count', 'dot_count',
                'bs_count', 'dash_count', 'url_entropy', 'params_count', 'subdomain_count',
                'domain_extension', 'underscores_count', 'questionmarks_count', 'equals_count', 'ampersands_count', 'digit_letter_ratio', 
                
                'pd_num_count', 'pd_non_alphanumeric_count', 
                
                'uppercase_dirs', 'path_count_special_chars',
                'path_uppercase_to_lowercase_ratio', 'params_length', 'queries_count']

dfX = pd.DataFrame(X.toarray(), columns=column_names)
dfX.to_csv("general-features.csv", index=False)