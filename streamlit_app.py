import streamlit as st
import requests

st.title("🧬 Drug Binding Predictor")

# --- input ---
smiles = st.text_input("SMILES")
protein = st.text_area("Protein sequence (optional)")

# --- выбор модели ---
model_type = st.selectbox(
    "Choose model",
    ["rf", "xgb", "dl", "gnn"]
)

# --- кнопка ---
if st.button("Predict"):

    if not smiles:
        st.error("Введите SMILES")
    else:
        try:
            res = requests.post(
                "http://127.0.0.1:8000/predict",
                json={
                    "smiles": smiles.strip(),
                    "protein": protein.strip() if protein else "",
                    "model_type": model_type
                },
                timeout=10
            )

            if res.status_code == 200:
                try:
                    data = res.json()
                except Exception:
                    st.error("Invalid response from API")
                    st.stop()

                st.success("Prediction done")

                st.subheader("Result")

                st.write(f"🧠 Model used: {data.get('model_used')}")
                st.write(f"💊 Predicted pKd: {data.get('predicted_pKd')}")

                st.json(data)

            else:
                st.error(f"API error: {res.text}")

        except Exception as e:
            st.error(f"Server not reachable: {e}")