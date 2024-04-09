import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, f_classif 
import numpy as np

X = pd.read_csv("datasets/feature_updated_dataset_X.csv")
y = pd.read_csv("datasets/feature_updated_dataset_y.csv")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=69)
y_train, y_val = y_train.values.ravel(), y_val.values.ravel()

model = RandomForestClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# Evaluation
accuracy = accuracy_score(y_val, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(y_val, y_pred, average='weighted')
print(f"Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | f1 score: {f1} | {model.__class__.__name__}")

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(y_val, y_pred)
for i in range(4):
    print(f"Class {i}:\nAccuracy: {accuracy[i]} | Precision: {precision[i]} | Recall: {recall[i]} | f1 score: {f1[i]} | {model.__class__.__name__}")
