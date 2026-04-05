import streamlit as st
import pandas as pd
import numpy as np

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

st.title("DDoS Detection System")

# Upload file
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:

    # =====================
    # LOAD DATA
    # =====================
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("Dataset Preview")
    st.write(df.head())

    # =====================
    # CHECK TARGET COLUMN
    # =====================
    if "Attack_Type" not in df.columns:
        st.error("Dataset must contain 'Attack_Type' column")
        st.stop()

    # =====================
    # FEATURE / TARGET SPLIT
    # =====================
    drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Label", "Attack_Type"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(drop_cols, axis=1)
    y = df["Attack_Type"]

    # =====================
    # DATA CLEANING
    # =====================
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    # Safe conversions
    if "Source Port" in X.columns:
        X["Source Port"] = pd.to_numeric(X["Source Port"], errors="coerce")

    if "SimillarHTTP" in X.columns:
        X["SimillarHTTP"] = X["SimillarHTTP"].astype(str)
        X["SimillarHTTP"] = X["SimillarHTTP"].map({
            "True": 1,
            "False": 0,
            "1": 1,
            "0": 0
        })

    X.fillna(0, inplace=True)

    # =====================
    # ENCODING TARGET
    # =====================
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # =====================
    # TRAIN TEST SPLIT
    # =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42
    )

    # =====================
    # RUN MODELS BUTTON
    # =====================
    if st.button("Run Model"):

        # =====================
        # RANDOM FOREST
        # =====================
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        st.subheader("Random Forest Results")
        st.write("Accuracy:", accuracy_score(y_test, rf_pred))
        st.text(classification_report(y_test, rf_pred))

        # =====================
        # XGBOOST
        # =====================
        xgb_tuned = XGBClassifier(
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.5,
            min_child_weight=2,
            reg_lambda=1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=len(le.classes_),
            tree_method="hist",
            random_state=42
        )

        xgb_tuned.fit(X_train, y_train)
        y_pred = xgb_tuned.predict(X_test)

        st.subheader("XGBoost Results")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))

        st.text(classification_report(
            le.inverse_transform(y_test),
            le.inverse_transform(y_pred)
        ))
