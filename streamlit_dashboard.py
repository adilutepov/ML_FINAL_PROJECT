import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="ML Drug Dashboard", layout="wide")

st.title("📊 ML + Drug Discovery Monitoring Dashboard")

# =========================
# MLflow LEADERBOARD
# =========================
@st.cache_data
def load_mlflow_runs():
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=["0"])

    data = []

    for run in runs:
        data.append({
            "model": run.data.params.get("model"),
            "rmse": run.data.metrics.get("rmse"),
            "mae": run.data.metrics.get("mae"),
            "r2": run.data.metrics.get("r2"),
        })

    return pd.DataFrame(data)


st.subheader("🏆 Model Leaderboard (MLflow)")

try:
    mlflow_df = load_mlflow_runs()

    if mlflow_df.empty:
        st.info("No MLflow runs yet")
    else:
        mlflow_df = mlflow_df.dropna()

        st.dataframe(mlflow_df.sort_values("rmse"))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.bar_chart(mlflow_df.set_index("model")["rmse"])

        with col2:
            st.bar_chart(mlflow_df.set_index("model")["mae"])

        with col3:
            st.bar_chart(mlflow_df.set_index("model")["r2"])

except Exception as e:
    st.warning(f"MLflow not available: {e}")


# =========================
# FASTAPI LOGS LOADER
# =========================
def load_logs():
    res = requests.get("http://127.0.0.1:8000/logs?limit=200")
    data = res.json()
    return pd.DataFrame(data["logs"])


# =========================
# MAIN BUTTON
# =========================
if st.button("🔄 Load logs"):

    df = load_logs()

    st.success(f"Loaded {len(df)} records")

    # =========================
    # RAW DATA
    # =========================
    st.subheader("📦 Raw logs")
    st.dataframe(df)

    # =========================
    # MODEL DISTRIBUTION
    # =========================
    st.subheader("📊 Model usage distribution")

    st.bar_chart(df["model_type"].value_counts())

    # =========================
    # PREDICTION DISTRIBUTION
    # =========================
    st.subheader("📈 Prediction distribution (pKd)")

    fig, ax = plt.subplots()
    ax.hist(df["prediction"], bins=30, color="skyblue")
    ax.set_xlabel("pKd")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # =========================
    # MODEL COMPARISON
    # =========================
    st.subheader("📊 Model performance comparison")

    if "model_type" in df.columns:

        stats = df.groupby("model_type")["prediction"].agg(["mean", "std", "count"])

        st.dataframe(stats)

        st.bar_chart(stats["mean"])

    # =========================
    # FILTER SECTION
    # =========================
    st.subheader("🔎 Filter by model")

    model_filter = st.selectbox(
        "Choose model",
        ["all"] + list(df["model_type"].unique())
    )

    if model_filter != "all":
        df_filtered = df[df["model_type"] == model_filter]
        st.dataframe(df_filtered)

    # =========================
    # STATS SUMMARY
    # =========================
    st.subheader("📌 Summary stats")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total predictions", len(df))
    col2.metric("Avg pKd", round(df["prediction"].mean(), 3))
    col3.metric("Std pKd", round(df["prediction"].std(), 3))


# =========================
# LIVE PREDICTION (OPTIONAL UI)
# =========================
st.divider()

st.subheader("🧪 Live Prediction")

smiles = st.text_input("SMILES")
protein = st.text_area("Protein sequence (optional)")
model_type = st.selectbox("Model", ["rf", "xgb", "dl"])

if st.button("Predict"):

    if not smiles:
        st.error("SMILES is required")
    else:
        try:
            res = requests.post(
                "http://127.0.0.1:8000/predict",
                json={
                    "smiles": smiles,
                    "protein": protein,
                    "model_type": model_type
                }
            )

            if res.status_code == 200:
                data = res.json()

                st.success("Prediction complete")

                st.metric("Predicted pKd", data["predicted_pKd"])
                st.write("Model used:", data["model_used"])

            else:
                st.error(res.text)

        except Exception as e:
            st.error(f"API error: {e}")