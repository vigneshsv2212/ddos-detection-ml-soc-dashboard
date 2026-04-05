import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

st.set_page_config(page_title="DDoS Detection Dashboard", layout="wide")

st.title("🛡️ DDoS Detection SOC Dashboard")

uploaded_file = st.file_uploader("Upload Network Flow Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    # =====================
    # LOAD DATA
    # =====================
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # =====================
    # DATA PREVIEW
    # =====================
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # =====================
    # SOC METRICS
    # =====================
    st.subheader("📈 Traffic Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Unique Attack Types", df["Attack_Type"].nunique())
    col3.metric("Features", df.shape[1])

    # =====================
    # ATTACK DISTRIBUTION
    # =====================
    st.subheader("🚨 Attack Distribution")
    attack_counts = df["Attack_Type"].value_counts()
    st.bar_chart(attack_counts)

    # =====================
    # PREPROCESSING
    # =====================
    if "Attack_Type" not in df.columns:
        st.error("Dataset must contain 'Attack_Type'")
        st.stop()

    drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Label", "Attack_Type"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(drop_cols, axis=1)
    y = df["Attack_Type"]

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

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
    # ENCODING + SPLIT
    # =====================
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # =====================
    # RUN MODELS
    # =====================
    if st.button("🚀 Run Detection Models"):

        st.header("🤖 Model Execution")

        with st.spinner("Training models..."):

            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)

            # XGBoost
            xgb = XGBClassifier(
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

            xgb.fit(X_train, y_train)
            xgb_pred = xgb.predict(X_test)

        # =====================
        # ACCURACY DASHBOARD
        # =====================
        st.subheader("📈 Model Performance")

        rf_acc = accuracy_score(y_test, rf_pred)
        xgb_acc = accuracy_score(y_test, xgb_pred)

        col1, col2 = st.columns(2)
        col1.metric("Random Forest Accuracy", round(rf_acc, 3))
        col2.metric("XGBoost Accuracy", round(xgb_acc, 3))

        if xgb_acc > rf_acc:
            st.success("🏆 XGBoost is performing better")
        else:
            st.success("🏆 Random Forest is performing better")

        # =====================
        # CONFUSION MATRIX
        # =====================
        st.subheader("🔍 Confusion Matrix (RF)")

        cm = confusion_matrix(y_test, rf_pred)

        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # =====================
        # CLASSIFICATION REPORT
        # =====================
        st.subheader("📋 Classification Report")

        report = classification_report(y_test, rf_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df)

        # =====================
        # WEAKEST CLASS
        # =====================
        worst_class = min(
            [k for k in report.keys() if k.isdigit()],
            key=lambda x: report[x]["recall"]
        )

        st.warning(
            f"⚠️ Weakest detected attack class: {le.inverse_transform([int(worst_class)])[0]}"
        )

        # =====================
        # ALERT TABLE (SOC STYLE)
        # =====================
        st.subheader("🚨 Detected Alerts")

        results_df = pd.DataFrame({
            "Actual": le.inverse_transform(y_test),
            "Predicted": le.inverse_transform(xgb_pred)
        })

        # Confidence score
        probs = xgb.predict_proba(X_test)
        results_df["Confidence"] = np.max(probs, axis=1)

        alerts = results_df[results_df["Actual"] != results_df["Predicted"]]

        st.dataframe(alerts.sort_values("Confidence", ascending=False).head(50))

        # =====================
        # DOWNLOAD ALERTS
        # =====================
        csv = alerts.to_csv(index=False).encode('utf-8')

        st.download_button(
            "⬇ Download Alerts",
            csv,
            "alerts.csv",
            "text/csv"
        )
