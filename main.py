# Data Manipulation and Visulaization Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Libraries needed for ML model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

RANDOM_STATE = 200

# Importing CSV and analyzing Data
credit = pd.read_csv('https://koenig-media.raywenderlich.com/uploads/2018/07/Credit.csv', index_col=0)
credit.head()

# One Hot Encoding Data to 1 and 0
credit_ = credit.copy()

credit_["Student"] = credit["Student"].map(lambda student: 1 if student == "Yes" else 0)
credit_["Married"] = credit["Married"].map(lambda married: 1 if married == "Yes" else 0)
credit_["Female"] = credit["Gender"].map(lambda gender: 1 if gender == "Female" else 0)
credit_.drop("Gender", axis=1, inplace=True)

credit_.head()

# Creating Dummies for Eye Color since more than two options
pd.get_dummies(credit_["Eye Color"])
eye_color = pd.get_dummies(credit["Eye Color"], drop_first=True)
credit_ = pd.concat([credit_.drop("Eye Color", axis=1), eye_color], axis=1)

# Creating the threshold for a "good" credit score which will be used to train data
credit_["Creditworthy"] = credit["Rating"].map(lambda val: 0 if val < 600 else 1)
credit_.drop("Rating", axis=1, inplace=True)

# Creating Train/Test Split and training data
X = credit_[["Age", "Education", "Student", "Married"]]
y = credit_["Creditworthy"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=RANDOM_STATE)

# Running Logisitic Regressor
estimator = LogisticRegression(random_state=RANDOM_STATE)
estimator.fit(X_train, y_train)

y_true = y_test  
y_pred = estimator.predict(X_test)  
y_score = estimator.predict_proba(X_test)[:, 0]

# Evaluation of Data
accuracy_score(y_true=y_true, y_pred=y_pred)
confusion_matrix(y_true=y_true, y_pred=y_pred)
precision_score(y_true=y_true, y_pred=y_pred)
recall_score(y_true=y_true, y_pred=y_pred)

# Iteration and Tuning of Parameters
X = credit_[["Income", "Education", "Student", "Married"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=RANDOM_STATE)
estimator = LogisticRegression(random_state=RANDOM_STATE)
estimator.fit(X_train, y_train)

y_true = y_test
y_pred = estimator.predict(X_test)
y_score = estimator.predict_proba(X_test)[:, 0]

# Evaluation of model with new parameters
accuracy_score(y_true=y_true, y_pred=y_pred)
confusion_matrix(y_true=y_true, y_pred=y_pred)
precision_score(y_true=y_true, y_pred=y_pred)
recall_score(y_true=y_true, y_pred=y_pred)
