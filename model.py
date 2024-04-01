import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

X = pd.read_csv("datasets/feature_updated_dataset_X.csv")
y = pd.read_csv("datasets/feature_updated_dataset_y.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
y_train, y_test = y_train.values.ravel(), y_test.values.ravel()

# Modelling
# 1) Logistic Regression
model_lr = LogisticRegression()

# 2) SVM
model_svm = svm.SVC()

# 3) Random Forest Classifier
model_rfc = RandomForestClassifier()

models = [model_lr, model_rfc, model_svm]
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Verifying model fit
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy} | F1 score: {f1} | {model.__class__.__name__}")