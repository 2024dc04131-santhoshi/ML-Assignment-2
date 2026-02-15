import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("Heart Disease Classification App")

st.sidebar.header("Select Model")

model_option = st.sidebar.selectbox(
    "Choose a Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV File", type=["csv"])
#you need to use encoded_heart.csv file from data/ folder
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if "target" not in data.columns:
        st.error("Uploaded dataset must contain 'target' column.")
    else:

        X = data.drop("target", axis=1)
        y = data["target"]

        # Load selected model
        model_dict = {
            "Logistic Regression": "model/logistic_regression.pkl",
            "Decision Tree": "model/decision_tree.pkl",
            "KNN": "model/knn.pkl",
            "Naive Bayes": "model/naive_bayes.pkl",
            "Random Forest": "model/random_forest.pkl",
            "XGBoost": "model/xgboost.pkl"
        }

        model = joblib.load(model_dict[model_option])

        # Load scaler (for LR and KNN)
        scaler = joblib.load("model/scaler.pkl")

        # Scale if needed
        if model_option in ["Logistic Regression", "KNN"]:
            X = scaler.transform(X)

        # Predictions
        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_prob)
        else:
            auc = "Not Available"

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        # Display metrics
        st.subheader("Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", round(accuracy, 3))
        col2.metric("Precision", round(precision, 3))
        col3.metric("Recall", round(recall, 3))

        col1.metric("F1 Score", round(f1, 3))
        col2.metric("MCC", round(mcc, 3))
        col3.metric("AUC", round(auc, 3) if auc != "Not Available" else auc)

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)
        st.write(cm)

else:
    st.info("Please upload a test CSV file to evaluate the model.")

