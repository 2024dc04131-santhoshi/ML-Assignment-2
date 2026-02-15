# 1️⃣ Imports
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# 2️⃣ Set working directory (IMPORTANT)
os.chdir("/home/cloud/Desktop/ML_Assignment_2")


# 3️⃣ Load Dataset
data = pd.read_csv("data/heart.csv") ##if you are not able to read the csv file using this code, you can go to data folder
##where you can find the above dataset.


# 4️⃣ Encoding categorical variables
data = pd.get_dummies(data, drop_first=True)


# 5️⃣ Separate features and target
X = data.drop("target", axis=1)
y = data["target"]


# 6️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 7️⃣ Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 8️⃣ Train Models

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)


# 9️⃣ Save Models
os.makedirs("model", exist_ok=True)

joblib.dump(log_reg, "model/logistic_regression.pkl")
joblib.dump(dt, "model/decision_tree.pkl")
joblib.dump(knn, "model/knn.pkl")
joblib.dump(nb, "model/naive_bayes.pkl")
joblib.dump(rf, "model/random_forest.pkl")
joblib.dump(xgb, "model/xgboost.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("All models saved successfully!")
