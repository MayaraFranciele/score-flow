from __future__ import annotations

import pickle
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from config import PROBLEMAS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "scoreflow_model.pkl"

app = Flask(__name__)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run first: python treinar.py"
        )
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@app.route("/")
def index():
    artifact = load_model()
    return render_template(
        "index.html",
        problemas=PROBLEMAS,
        threshold=artifact["threshold"],
        best_model=artifact["best_model_name"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    artifact = load_model()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    threshold = artifact["threshold"]

    payload = request.form.to_dict()
    row = {}

    for col in feature_columns:
        if col in payload:
            value = payload[col]
            if value == "":
                row[col] = None
            elif value.lower() in {"sim", "true", "yes"}:
                row[col] = True
            elif value.lower() in {"nao", "false", "no"}:
                row[col] = False
            else:
                try:
                    row[col] = float(value)
                except ValueError:
                    row[col] = value

    # Garante dados faltantes e mantém ordem de features
    sample = {}
    for col in feature_columns:
        sample[col] = row.get(col, None)

    # Prevent division issues on income commitment
    if "comprometimento_renda" in sample and sample["comprometimento_renda"] is not None:
        sample["comprometimento_renda"] = max(0.0, float(sample["comprometimento_renda"]))

    import pandas as pd
    df = pd.DataFrame([sample], columns=feature_columns)
    proba = model.predict_proba(df)[0, 1]
    risk_level = "High" if proba >= threshold else "Low"

    return jsonify({
        "risco": risk_level,
        "probabilidade": round(float(proba), 4),
        "limiar": round(float(threshold), 4),
        "mensagem": "High default risk" if risk_level == "High" else "Low default risk",
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
