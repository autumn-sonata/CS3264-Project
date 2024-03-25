import pandas as pd

df = pd.read_csv("datasets/malicious_phish.csv")

# Check empty urls
empty_url_counter = 0

def check_empty_url(url):
    if len(url) == 0:
        empty_url_counter += 1

df['url'].apply(check_empty_url)
print(f"Empty URLs found: {empty_url_counter}\n")

# Check number of entries per type
print(f"Breakdown of numbers of entries per type: {df['type'].value_counts()}")