import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif 
import numpy as np

X = pd.read_csv("datasets/feature_updated_dataset_X.csv")
y = pd.read_csv("datasets/feature_updated_dataset_y.csv")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=69)
y_train, y_val = y_train.values.ravel(), y_val.values.ravel()

# Modelling
# 1) Logistic Regression
model_lr = LogisticRegression()

# 2) SVM
model_svm = svm.SVC()

# 3) Random Forest Classifier
model_rfc = RandomForestClassifier()

models = [model_rfc]
for model in models:
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    # # Verifying model fit
    # accuracy = accuracy_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print(f"Accuracy: {accuracy} | F1 score: {f1} | {model.__class__.__name__}")

    if model.__class__.__name__ == "RandomForestClassifier":
        feature_name_ls = ["csr_http", "csr_https", "csr_www", "csr_url_len", "csr_digit_count", 
      "csr_percentage_count", "csr_dot_count", "csr_bs_count", "csr_dash_count", 
      "csr_url_entropy", "csr_url_num_params", "csr_url_num_subdomains", "csr_domain_extension"]
        feature_set = set()
        for i in range(1, X_train.shape[1] + 1):
            model = RandomForestClassifier()
            select_feature_model = SelectKBest(f_classif, k=i)
            X_train_partition = select_feature_model.fit_transform(X_train, y_train)
            model.fit(X_train_partition, y_train)
            X_test_partition = select_feature_model.transform(X_val)
            y_pred = model.predict(X_test_partition)

            feature_importances = model.feature_importances_
            indices = np.argsort(feature_importances)[::-1]
            curr_feature_set = set()
            for j in indices:
                curr_feature_set.add(feature_name_ls[j])
            new_feature = curr_feature_set - feature_set
            feature_set = curr_feature_set

            # Verifying model fit
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            print(f"Accuracy: {accuracy} | F1 score: {f1} | {model.__class__.__name__} | Number of features: {i} | Added: {new_feature} | Features: {curr_feature_set}")
