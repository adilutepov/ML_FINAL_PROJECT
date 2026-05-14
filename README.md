# AI Drug-Target Binding Affinity Prediction 🧬💻

Welcome to the Drug-Target Binding Affinity (pKd) prediction project. This project utilizes classical Machine Learning algorithms, Deep Learning, and Graph Neural Networks (GNNs) to analyze drug molecules (SMILES) and protein sequences.

## 📂 Project Structure
- `ML_Final_Ntbk_Final.ipynb` — The core Jupyter Notebook containing data preprocessing, model training (RF, XGBoost, DL, GNN, Dual Encoder), and evaluation visualizations.
- `main.py` — The Backend API built with FastAPI, providing the prediction endpoints.
- `streamlit_app.py` — The Frontend web interface built with Streamlit for seamless user interaction with the models.
- `streamlit_dashboard.py` — An Analytics Dashboard displaying model metrics fetched directly from the MLflow registry.
- `gnn_model.py` / `protein_transformer.py` / `graph_builder.py` — Helper scripts containing the neural network architectures (GCN and Protein Transformer) and graph processing logic.

## 🚀 How to Run the Project

Before starting, ensure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

For the complete system to work, you need to run three independent services (in three separate terminal windows):

### 1. Start the Backend API (FastAPI)
The backend handles real-time predictions and runs the neural networks.
```bash
uvicorn main:app --reload
```
The API documentation will be available at: http://localhost:8000/docs

### 2. Start the User Interface (Streamlit)
The web interface for inputting SMILES and protein sequences.
```bash
streamlit run streamlit_app.py
```
The interface will open at: http://localhost:8501

### 3. Start the Analytics Dashboard (MLflow Dashboard)
A dashboard that displays a dynamic leaderboard of model metrics pulled straight from MLflow.
```bash
streamlit run streamlit_dashboard.py
```
The dashboard will open at: http://localhost:8502 (if the previous port is occupied).

## 🏆 MLflow Model Registry
This project implements a complete MLOps workflow. Models are logged, tracked, and natively registered in a local `mlflow.db`. 
To access the official MLflow UI, run:
```bash
mlflow ui
```
Navigate to http://localhost:5000 to view the registered models and their lifecycle stages (Production / Staging).
