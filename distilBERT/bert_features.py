import tldextract
import numpy as np
import pandas as pd

# Exploration into distilbbert features engineering splitting url, domain, subdomain, and suffix
df = pd.read_csv("datasets/malicious_phish.csv")

def preprocess_url(df):
    domains = []
    subdomains = []
    suffixes = []
    
    for url in df['url']:
        extracted = tldextract.extract(url)
        
        domain = extracted.domain.lower() if extracted.domain else 'na'
        subdomain = extracted.subdomain.lower() if extracted.subdomain else 'na'
        suffix = extracted.suffix.lower() if extracted.suffix else 'na'
        
        domains.append(domain)
        subdomains.append(subdomain)
        suffixes.append(suffix)
    
    
    df_features = pd.DataFrame({
    'url': df['url'],
    'domain': domains,
    'subdomain': subdomains,
    'suffix': suffixes
    })
    
    return df_features

preprocessed_df = preprocess_url(df)
preprocessed_df.to_csv("datasets/bert_features_dataset_X.csv", index=False)
