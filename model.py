import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif 
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

X = pd.read_csv("datasets/feature_updated_dataset_X_reduced.csv")
y = pd.read_csv("datasets/feature_updated_dataset_y.csv")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=69)
y_train, y_val = y_train.values.ravel(), y_val.values.ravel()

scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Modelling
# 1) Logistic Regression
model_lr = LogisticRegression()

# 2) SVM
model_svm = svm.SVC()

# 3) Random Forest Classifier
model_rfc = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=69)

models = [model_rfc]
for model in models:
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    # # Verifying model fit
    # accuracy = accuracy_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print(f"Accuracy: {accuracy} | F1 score: {f1} | {model.__class__.__name__}")

    if model.__class__.__name__ == "RandomForestClassifier":
        column_names = ['www', 'url_length', 'digit_count', 'percentage_count', 
                'bs_count', 'dash_count', 'url_entropy', 'params_count', 
                'domain_extension', 'underscores_count', 'equals_count', 'ampersands_count', 'digit_letter_ratio', 
                
                'pd_num_count', 'pd_at_count', 'pd_hyphen_count', 'pd_in_alex_top_1m']
        feature_set = set()
        for i in range(1, X_train.shape[1] + 1):
            model = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=69)
            select_feature_model = SelectKBest(mutual_info_classif, k=i)
            X_train_partition = select_feature_model.fit_transform(X_train, y_train)
            model.fit(X_train_partition, y_train)
            X_test_partition = select_feature_model.transform(X_val)
            y_pred = model.predict(X_test_partition)

            feature_importances = model.feature_importances_
            indices = np.argsort(feature_importances)[::-1]
            curr_feature_set = set()
            for j in indices:
                curr_feature_set.add(column_names[j])
            new_feature = curr_feature_set - feature_set
            feature_set = curr_feature_set

            # Verifying model fit
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            print(f"Accuracy: {accuracy} | F1 score: {f1} | {model.__class__.__name__} | Number of features: {i} | Added: {new_feature} | Features: {curr_feature_set}")
