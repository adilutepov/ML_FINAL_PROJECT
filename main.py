from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import numpy as np
from tensorflow import keras

import sqlite3
from datetime import datetime
from datetime import datetime, timezone

app = FastAPI(
    title="AI Drug Binding Prediction API",
    description="Predict pKd from SMILES using RF / XGBoost / DL",
    version="1.0"
)

# --- CORS (для Streamlit) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- загрузка Векторов ---
smiles_vec = joblib.load("smiles_vec.joblib")
protein_vec = joblib.load("protein_vec.joblib")

# --- загрузка моделей ---

rf_model = joblib.load("rf_model.joblib")
xgb_model = joblib.load("xgb_model.joblib")

dl_model = keras.models.load_model("dl_model.keras")

import torch
from gnn_model import GCN
import os

gnn_model = None
if os.path.exists("gnn_model.pth"):
    device = torch.device("cpu")
    gnn_model = GCN().to(device)
    gnn_model.load_state_dict(torch.load("gnn_model.pth", map_location=device, weights_only=True))
    gnn_model.eval()
# --- таблица ---

conn = sqlite3.connect("predictions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    smiles TEXT,
    protein TEXT,
    model_type TEXT,
    prediction REAL
)
""")

conn.commit()


# --- input schema ---
from pydantic import BaseModel, Field

class DrugRequest(BaseModel):
    smiles: str = Field(..., min_length=1, description="SMILES string of compound")
    protein: str | None = Field(None, description="Protein sequence (optional)")
    model_type: str = Field("rf", description="rf / xgb / dl / gnn")


# --- health check ---
@app.get("/")
def root():
    return {"message": "AI Drug Prediction API is running"}

@app.get("/logs")
def get_logs(limit: int = 50):

    cursor.execute("""
        SELECT timestamp, smiles, protein, model_type, prediction
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()

    return {
        "count": len(rows),
        "logs": [
            {
                "timestamp": r[0],
                "smiles": r[1],
                "protein": r[2],
                "model_type": r[3],
                "prediction": r[4]
            }
            for r in rows
        ]
    }



# --- prediction endpoint ---
@app.post("/predict")
def predict(data: DrugRequest):

    if not data.smiles or len(data.smiles.strip()) == 0:
        return {"error": "SMILES cannot be empty"}

    # --- transform SMILES ---
    X_smiles = smiles_vec.transform([data.smiles])

    # --- transform protein ---
    protein_input = data.protein.strip() if data.protein else ""
    X_protein = protein_vec.transform([protein_input])

    # --- combine features ---
    from scipy.sparse import hstack
    X = hstack([X_smiles, X_protein])

    # --- model selection ---
    if data.model_type == "rf":
        pred = rf_model.predict(X)[0]

    elif data.model_type == "xgb":
        pred = xgb_model.predict(X)[0]

    elif data.model_type == "dl":
        X_dense = X.toarray()
        pred = dl_model.predict(X_dense).flatten()[0]
        
    elif data.model_type == "gnn" and gnn_model is not None:
        from graph_builder import smiles_to_graph
        graph = smiles_to_graph(data.smiles)
        if graph is None:
            return {"error": "Invalid SMILES for GNN"}
        x, edge_index = graph
        batch = torch.zeros(x.shape[0], dtype=torch.long)
        with torch.no_grad():
            pred = gnn_model(x, edge_index, batch).item()
    elif data.model_type == "gnn" and gnn_model is None:
        return {"error": "GNN model file not found. Please train it first."}

    valid_models = {"rf", "xgb", "dl", "gnn"}

    if data.model_type not in valid_models:
        return {
        "error": f"Invalid model_type: {data.model_type}",
        "valid_options": list(valid_models)
    }


    cursor.execute("""
    INSERT INTO predictions (timestamp, smiles, protein, model_type, prediction)
    VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        data.smiles,
        data.protein,
        data.model_type,
        float(pred)
    ))

    conn.commit()
       

    # --- response ---
    return {
        "smiles": data.smiles,
        "model_used": data.model_type,
        "protein_used": data.protein is not None,
        "predicted_pKd": float(pred)
    }