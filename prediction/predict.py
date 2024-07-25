import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings
import sys
import pickle
import subprocess

warnings.simplefilter("ignore")

models = ["lexical", "general", "minimal"]
model_pairs = []
url = sys.argv[1]
predictions = {}
url_features = {}
for model_name in models:
    # Extracting url features based on model
    process = subprocess.Popen("python3 " + model_name + "-features.py " + url, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for extraction finish
    return_code = process.wait()

    if (model_name == "lexical"):
        features = pd.read_csv(model_name + '-features.csv', header=None, skiprows=1)
    else:
        features = pd.read_csv(model_name + '-features.csv')

    model_file_path = "rf-" + model_name + ".pkl"
    with open(model_file_path, "rb") as file:
        model = pickle.load(file)
        model_pairs.append((file.name, model))
        url_features[file.name] = features

    predictions[model] = model.predict_proba(features)
    
y_ensemble_pred = None # holds predictions of all models
model_col = {} # Takes note of which column the model predictions are in <index, model_name>
model_col_rev = {} # <model_name, index>

for i in range(len(model_pairs)):
    model_name, model = model_pairs[i]
    X_test = url_features[model_name] # Already scaled
    
    y_pred = model.predict(X_test)
    if y_ensemble_pred is None:
        y_ensemble_pred = np.empty((y_pred.shape[0], 3))

    y_ensemble_pred[:, i] = y_pred
    model_col[i] = model_name
    model_col_rev[model_name] = i
    
pred_combined_y = y_ensemble_pred[:, model_col_rev["rf-general.pkl"]]

row = 0
for entry in y_ensemble_pred:
    index_benign_spec = model_col_rev["rf-minimal.pkl"]
    index_lexical = model_col_rev["rf-lexical.pkl"]
    if entry[index_benign_spec] == 0 and entry[index_lexical] == 0:
        pred_combined_y[row] = 0
    row += 1
    
prediction = pred_combined_y[0]
if (prediction == 0):
    print("benign")
elif (prediction == 1):
    print("defacement")
elif (prediction == 2):
    print("phishing")
else:
    print("malware")