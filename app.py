"""
Telecom Customer Churn Prediction - Streamlit App
Run with:  streamlit run app.py

Model loading order:
  1. telecom_churn_logistic_pipeline.pkl  (saved from the notebook)  <- preferred
  2. If the .pkl is missing, the app trains the same pipeline from Telco_Customer_Churn.csv
     (in the same folder) and saves the .pkl for next time.
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Config (mirrors the notebook)
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
MODEL_PATH = "telecom_churn_logistic_pipeline.pkl"
DATA_PATH = "Telco_Customer_Churn.csv"

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "NumAddOnServices", "HasInternetService", "HasFamily",
]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "TenureGroup", "MonthlyChargeCategory",
]
ADDON_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
RAW_INPUT_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
]


# ----------------------------------------------------------------------------
# Feature engineering (same logic as notebook cells 8-12)
# ----------------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["NumAddOnServices"] = (df[ADDON_COLUMNS] == "Yes").sum(axis=1)

    df["HasInternetService"] = np.where(df["InternetService"] == "No", 0, 1)

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New", "Medium", "Long", "Very Long"],
    )

    df["MonthlyChargeCategory"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 70, 100, 120],
        labels=["Low", "Medium", "High", "Very High"],
    )

    df["HasFamily"] = np.where(
        (df["Partner"] == "Yes") | (df["Dependents"] == "Yes"), 1, 0
    )
    return df


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning so uploaded CSVs behave like the notebook data."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen must be 0/1 integers (as in the original dataset)
    if df["SeniorCitizen"].dtype == object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map(
            {"Yes": 1, "No": 0, "1": 1, "0": 0}
        )
    df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)
    return df


# ----------------------------------------------------------------------------
# Model loading / training
# ----------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        )),
    ])


def train_from_csv(path: str) -> Pipeline:
    data = pd.read_csv(path)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna(subset=["TotalCharges"]).drop_duplicates().copy()
    data = add_engineered_features(data)

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = data["Churn"].map({"No": 0, "Yes": 1})

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_PATH)
    return pipe


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    if os.path.exists(DATA_PATH):
        return train_from_csv(DATA_PATH)
    return None


def predict(model, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return raw_df with churn probability and label appended."""
    prepared = add_engineered_features(clean_raw_dataframe(raw_df))
    X = prepared[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    proba = model.predict_proba(X)[:, 1]
    return prepared, proba


def risk_band(p: float) -> str:
    if p >= 0.70:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡", layout="wide")
st.title("📡 Telecom Customer Churn Predictor")
st.caption(
    "Logistic Regression pipeline (balanced class weights) trained on the "
    "Telco Customer Churn dataset."
)

model = load_model()

if model is None:
    st.error(
        f"Could not find `{MODEL_PATH}` or `{DATA_PATH}` in the app folder.\n\n"
        "Put one of these files next to `app.py` and refresh:\n"
        f"- `{MODEL_PATH}` (created by the last cells of your notebook), or\n"
        f"- `{DATA_PATH}` (the app will train the model for you)."
    )
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Churn decision threshold", 0.05, 0.95, 0.50, 0.05,
        help="Customers with churn probability >= threshold are labelled 'Will churn'. "
             "Lower it to catch more churners (higher recall, lower precision).",
    )
    st.markdown("---")
    st.markdown(
        "**Notebook test-set results (Logistic Regression)**\n\n"
        "- Accuracy: 0.736\n- Precision: 0.503\n- Recall: 0.791\n"
        "- F1: 0.615\n- ROC-AUC: 0.837"
    )

tab_single, tab_batch = st.tabs(["🧍 Single customer", "📄 Batch (CSV upload)"])

# ---------------------------- Single prediction -----------------------------
with tab_single:
    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

            st.subheader("Account")
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)",
                ],
            )

        with c2:
            st.subheader("Phone & Internet")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox(
                "Multiple lines", ["No", "Yes", "No phone service"]
            )
            internet = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )

            st.subheader("Add-on services")
            no_net = internet == "No"
            addon_opts = ["No internet service"] if no_net else ["No", "Yes"]
            online_sec = st.selectbox("Online security", addon_opts)
            online_bkp = st.selectbox("Online backup", addon_opts)
            dev_prot = st.selectbox("Device protection", addon_opts)

        with c3:
            st.subheader("More add-ons")
            tech = st.selectbox("Tech support", addon_opts)
            stream_tv = st.selectbox("Streaming TV", addon_opts)
            stream_mv = st.selectbox("Streaming movies", addon_opts)

            st.subheader("Charges")
            monthly = st.number_input(
                "Monthly charges", min_value=18.0, max_value=120.0,
                value=70.0, step=0.5,
            )
            default_total = float(round(monthly * tenure, 2))
            total = st.number_input(
                "Total charges (defaults to monthly × tenure)",
                min_value=0.0, max_value=15000.0,
                value=default_total, step=10.0,
            )

        submitted = st.form_submit_button("🔮 Predict churn", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bkp,
            "DeviceProtection": dev_prot,
            "TechSupport": tech,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_mv,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }])

        prepared, proba = predict(model, row)
        p = float(proba[0])
        will_churn = p >= threshold

        st.markdown("### Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{p:.1%}")
        m2.metric("Prediction", "Will churn" if will_churn else "Will stay")
        m3.metric("Risk level", risk_band(p))
        st.progress(min(max(p, 0.0), 1.0))

        if will_churn:
            st.error(
                "⚠️ This customer is likely to churn. Consider a retention offer, "
                "a longer-term contract incentive, or add-on bundles."
            )
        else:
            st.success("✅ This customer is likely to stay.")

        with st.expander("Engineered features used by the model"):
            st.dataframe(
                prepared[["NumAddOnServices", "HasInternetService", "HasFamily",
                          "TenureGroup", "MonthlyChargeCategory"]],
                hide_index=True,
            )

# ---------------------------- Batch prediction ------------------------------
with tab_batch:
    st.markdown(
        "Upload a CSV with the same columns as the Telco dataset "
        "(`customerID` and `Churn` are optional)."
    )
    st.code(", ".join(RAW_INPUT_COLUMNS), language="text")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            missing = [c for c in RAW_INPUT_COLUMNS if c not in raw.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                prepared, proba = predict(model, raw[RAW_INPUT_COLUMNS])

                out = raw.copy()
                out["Churn_Probability"] = proba.round(4)
                out["Predicted_Churn"] = np.where(proba >= threshold, "Yes", "No")
                out["Risk_Level"] = [risk_band(p) for p in proba]

                n = len(out)
                n_churn = int((out["Predicted_Churn"] == "Yes").sum())
                k1, k2, k3 = st.columns(3)
                k1.metric("Customers", f"{n:,}")
                k2.metric("Predicted to churn", f"{n_churn:,}")
                k3.metric("Predicted churn rate", f"{n_churn / n:.1%}")

                st.bar_chart(out["Risk_Level"].value_counts().reindex(
                    ["Low", "Medium", "High"]).fillna(0))

                st.dataframe(
                    out.sort_values("Churn_Probability", ascending=False),
                    use_container_width=True,
                )
                st.download_button(
                    "⬇️ Download predictions (CSV)",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not process the file: {e}")