import math
import pandas as pd
from sklearn.ensemble import VotingClassifier
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re

domain_index_df = pd.read_csv('datasets\domain_index_mapping.csv')

domain_to_index = dict(zip(domain_index_df['Domain'], domain_index_df['Index']))

def get_entropy(text):
  text_strip = text.strip()
  prob = [float(text_strip.count(c)) / len(text_strip) for c in dict.fromkeys(list(text_strip))]
  text_entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
  return text_entropy

def get_num_subdomains(text):
  if 'http' in text:
    num_subdomains = text.split('http')[-1].split('//')[-1].split('/')
  else:
    num_subdomains = text.split('https')[-1].split('//')[-1].split('/')
  num_subdomains = len(num_subdomains)-1
  return num_subdomains

def get_domain_extension(text):
  domain_extension = text.split('.')[-1].split('/')[0]
  return domain_to_index.get(domain_extension, len(domain_to_index) + 1)

def extract_features(text):
  rf_input = ['http', 'https', 'www', 'url_length', 'digit_count', 'percentage_count',
       'dot_count', 'bs_count', 'dash_count', 'url_entropy', 'params_count',
       'subdomains_count', 'domain_extension']
  df = pd.DataFrame(columns=rf_input)
  
  http_count = len(re.findall(r"http://", text))
  https_count = len(re.findall(r"https://", text))
  www_count = len(re.findall(r"www", text))
  url_length = len(text)
  num_digits = len(re.findall(r"\d", text))
  num_percentages = len(re.findall(r"%", text))
  num_dots = len(re.findall(r"\.", text))
  num_slashes = len(re.findall(r"/", text))
  num_dash = len(re.findall(r"-", text))
  text_entropy = get_entropy(text)
  num_params = len(text.split('&')) - 1
  num_subdomains = get_num_subdomains(text)
  url_domain = get_domain_extension(text)
  data = {'http': [http_count], 'https': [https_count], 'www': [www_count], 'url_length': [url_length], 
          'digit_count': [num_digits], 'percentage_count': [num_percentages], 'dot_count': [num_dots], 
          'bs_count': [num_slashes], 'dash_count': [num_dash], 'url_entropy': [text_entropy], 
          'params_count': [num_params], 'subdomains_count': [num_subdomains], 'domain_extension': [url_domain]}
  df = pd.DataFrame.from_dict(data)  
  return df

# # Load the BERT model and tokenizer
# distilbert_model = AutoModelForSequenceClassification.from_pretrained('bert_model_url_classification\distilBERT pretrain split then undersampled uncased')
# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
# distilbert_model.eval()

# Load the RF model
rf_model = joblib.load('model_rfc1.pkl')

# # Create the ensemble classifier
# ensemble_classifier = VotingClassifier(estimators=[
#   ('distilbert', distilbert_model),
#   ('rf', rf_model)
# ])

# # Use the ensemble classifier for prediction
# input_text = "This is a sample input"
# input_tokens = tokenizer.encode_plus(input_text, add_special_tokens=True, return_tensors='pt')
# predictions = ensemble_classifier.predict(input_tokens['input_ids'])

# print(predictions)

while True:
  # Take user input
  input_text = extract_features(input("Enter a text: "))
  # Make prediction using the RF model
  prediction = rf_model.predict(input_text)

  # Print the prediction
  print(prediction)