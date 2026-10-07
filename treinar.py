from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "credito.csv"
MODEL_PATH = BASE_DIR / "scoreflow_model.pkl"


def gerar_dataset_semente() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 6000
    finalidade = ["educacao", "veiculo", "casa", "consumo", "negocios", "outros"]
    estado_civil = ["solteiro", "casado", "divorciado", "viuvo"]
    grau_instrucao = ["fundamental", "medio", "superior", "pos"]
    tipo_emprego = ["clt", "autonomo", "temporario", "estudante"]
    regiao = ["norte", "sul", "sudeste", "nordeste", "centro_oeste"]
    posse_imovel = ["sim", "nao"]

    idade = rng.integers(18, 75, size=n)
    renda_mensal = rng.normal(4800, 1800, size=n)
    renda_mensal = np.clip(renda_mensal, 1200, 18000)

    valor_emprestimo = rng.uniform(2500, 48000, size=n)
    prazo_meses = rng.integers(6, 72, size=n)
    taxa_juros = rng.uniform(4.0, 28.0, size=n)
    score_credito = rng.integers(420, 820, size=n)
    valor_prestacao = rng.uniform(180, 2500, size=n)
    tempo_emprego = rng.integers(0, 18, size=n)
    dependentes = rng.integers(0, 5, size=n)
    divida_total = rng.uniform(0, 35000, size=n)
    gasto_mensal = rng.uniform(500, 9000, size=n)
    saldo_mensal = rng.uniform(-1800, 6000, size=n)
    valor_bem = rng.uniform(0, 220000, size=n)

    finalidade_col = rng.choice(finalidade, size=n)
    posse_imovel_col = rng.choice(posse_imovel, size=n)
    estado_civil_col = rng.choice(estado_civil, size=n)
    grau_instrucao_col = rng.choice(grau_instrucao, size=n)
    tipo_emprego_col = rng.choice(tipo_emprego, size=n)
    regiao_col = rng.choice(regiao, size=n)

    # Proporção de inadimplência esperada: ~21%
    risco_latente = (
        -1.7
        + 0.025 * ((score_credito - 650) / 10)
        + 0.0008 * (valor_emprestimo / np.maximum(1, renda_mensal))
        + 0.4 * (valor_prestacao / np.maximum(renda_mensal, 1))
        + 0.25 * (divida_total / np.maximum(renda_mensal, 1))
        - 0.45 * (posse_imovel_col == "sim")
        + 0.3 * (finalidade_col == "consumo")
        + 0.2 * (estado_civil_col == "solteiro")
        + 0.08 * (tipo_emprego_col == "autonomo")
    )
    prob = 1 / (1 + np.exp(-risco_latente))
    inadimplente = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "id_contrato": np.arange(1, n + 1),
        "idade": idade,
        "renda_mensal": renda_mensal.round(2),
        "valor_emprestimo": valor_emprestimo.round(2),
        "prazo_meses": prazo_meses,
        "taxa_juros": taxa_juros.round(2),
        "score_credito": score_credito,
        "valor_prestacao": valor_prestacao.round(2),
        "tempo_emprego": tempo_emprego,
        "dependentes": dependentes,
        "divida_total": divida_total.round(2),
        "gasto_mensal": gasto_mensal.round(2),
        "saldo_mensal": saldo_mensal.round(2),
        "valor_bem": valor_bem.round(2),
        "finalidade": finalidade_col,
        "posse_imovel": posse_imovel_col,
        "estado_civil": estado_civil_col,
        "grau_instrucao": grau_instrucao_col,
        "tipo_emprego": tipo_emprego_col,
        "regiao": regiao_col,
        "inadimplente": inadimplente,
    })

    # Ajuste fino para manter taxa aproximada 21%
    alvo = df["inadimplente"].mean()
    if alvo > 0.25:
        idx_default = df.index[df["inadimplente"] == 1]
        n_to_adjust = max(0, int(round((alvo - 0.21) * len(df))))
        if len(idx_default) > 0 and n_to_adjust > 0:
            to_flip = rng.choice(idx_default, size=min(n_to_adjust, len(idx_default)), replace=False)
            df.loc[to_flip, "inadimplente"] = 0

    df["inadimplente"] = df["inadimplente"].astype(int)
    return df


def ensure_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        df = gerar_dataset_semente()
        df.to_csv(DATA_PATH, index=False)
        print(f"Arquivo gerado em {DATA_PATH} com {len(df)} linhas.")


def normalize_target(series: pd.Series) -> pd.Series:
    s = series.copy().astype(str).str.strip().str.lower()
    mapping = {
        "0": 0,
        "1": 1,
        "nao": 0,
        "não": 0,
        "adimplente": 0,
        "adimplencia": 0,
        "sim": 1,
        "inadimplente": 1,
        "default": 1,
        "true": 1,
        "false": 0,
    }
    mapped = s.map(mapping)
    if mapped.isna().any():
        numeric = pd.to_numeric(s, errors="coerce")
        mapped = mapped.fillna(numeric)
    return mapped.fillna(0).astype(int)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {
        str(col).lower().replace(" ", "_").replace("-", "_").replace(".", "_"): col
        for col in df.columns
    }
    for candidate in candidates:
        key = candidate.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        if key in normalized:
            return normalized[key]
    return None


def plot_target_distribution(y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    labels = ["Adimplente", "Inadimplente"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=values, palette=["#2E8B57", "#C0392B"])
    plt.title("Proporção de classes (Alvo)")
    plt.ylabel("Quantidade")
    plt.xlabel("Classe")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "eda_proporcao_classes.png", dpi=200)
    plt.close()


def plot_nulls(df: pd.DataFrame) -> None:
    nulls = df.isnull().sum().sort_values(ascending=False)
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("Nenhuma coluna com valores nulos.")
        return

    plt.figure(figsize=(10, 5))
    sns.barplot(x=nulls.index, y=nulls.values, palette="viridis")
    plt.title("Valores nulos por coluna")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "eda_nulos.png", dpi=200)
    plt.close()


def plot_inadimplencia_vs_score(df: pd.DataFrame, target_col: str, score_col: str) -> None:
    if not score_col or score_col not in df.columns:
        return
    plot_df = df[[score_col, target_col]].copy().dropna(subset=[score_col])
    plot_df["score_bin"] = pd.qcut(plot_df[score_col], q=10, duplicates="drop")
    grouped = plot_df.groupby("score_bin", observed=False)[target_col].mean().reset_index()
    grouped["score_bin"] = grouped["score_bin"].astype(str)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=grouped, x="score_bin", y=target_col, palette="rocket")
    plt.title("Inadimplência por faixa de score")
    plt.xlabel("Faixa de score")
    plt.ylabel("Taxa de inadimplência")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "eda_inadimplencia_score.png", dpi=200)
    plt.close()


def plot_inadimplencia_por_categoria(df: pd.DataFrame, target_col: str, col_name: str, title: str, output_name: str) -> None:
    if not col_name or col_name not in df.columns:
        return
    grouped = df.groupby(col_name, dropna=False)[target_col].mean().reset_index()
    grouped.columns = [col_name, "taxa_inadimplencia"]

    plt.figure(figsize=(8, 5))
    sns.barplot(data=grouped, x=col_name, y="taxa_inadimplencia", palette="Set2")
    plt.title(title)
    plt.ylabel("Taxa de inadimplência")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(BASE_DIR / output_name, dpi=200)
    plt.close()


def create_comprometimento_renda(df: pd.DataFrame) -> pd.DataFrame:
    for renda_col, parcela_col in [
        ("renda_mensal", "valor_prestacao"),
        ("renda_mensal", "valor_emprestimo"),
    ]:
        if renda_col in df.columns and parcela_col in df.columns:
            renda = df[renda_col].replace(0, np.nan)
            parcela = df[parcela_col]
            df["comprometimento_renda"] = parcela / renda
            return df
    df["comprometimento_renda"] = 0.0
    return df


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("Nenhuma coluna foi detectada para treino.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_variants() -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }


def build_param_grid(model_name: str) -> dict:
    if model_name == "LogisticRegression":
        return {
            "model__C": [0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            "model__solver": ["liblinear", "lbfgs"],
            "model__class_weight": [None, "balanced"],
        }
    if model_name == "RandomForest":
        return {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 8, 12, 16],
            "model__min_samples_leaf": [1, 3, 5],
            "model__class_weight": [None, "balanced"],
        }
    if model_name == "GradientBoosting":
        return {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0],
        }
    return {}


def get_financial_threshold(y_true: np.ndarray, y_proba: np.ndarray, fp_cost: float = 1500.0, fn_cost: float = 8000.0) -> tuple[float, float]:
    best_threshold = 0.5
    best_cost = np.inf

    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total_cost = fp * fp_cost + fn * fn_cost
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = threshold
    return float(best_threshold), float(best_cost)


def evaluate_thresholds_by_cost(y_true: np.ndarray, y_proba: np.ndarray, thresholds: list[float] | None = None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.15, 0.85, 15).tolist()

    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        total_cost = fp * 1500.0 + fn * 8000.0
        rows.append({
            "threshold": round(float(threshold), 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "custo_total": float(total_cost),
        })

    return pd.DataFrame(rows).sort_values("custo_total").reset_index(drop=True)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: Path, title: str = "Curva ROC") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#3b82f6", linewidth=2)
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Taxa de falso positivo")
    plt.ylabel("Taxa de verdadeiro positivo")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compare_models_cv(X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold) -> pd.DataFrame:
    results = []
    model_variants = build_model_variants()
    preprocessor = build_preprocessor(
        list(X_train.select_dtypes(include=["number"]).columns),
        list(X_train.select_dtypes(exclude=["number"]).columns),
    )

    for name, model in model_variants.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        results.append({
            "modelo": name,
            "roc_auc_medio": float(np.mean(scores)),
            "desvio_padrao": float(np.std(scores)),
            "folds": scores.tolist(),
        })

    metrics_df = pd.DataFrame(results).sort_values("roc_auc_medio", ascending=False).reset_index(drop=True)
    print("\nTabela de comparação por validação cruzada:")
    print(metrics_df[["modelo", "roc_auc_medio", "desvio_padrao"]].to_string(index=False))
    return metrics_df


def compare_class_weight_effect(X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold) -> pd.DataFrame:
    rows = []
    for class_weight in [None, "balanced"]:
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(
                list(X_train.select_dtypes(include=["number"]).columns),
                list(X_train.select_dtypes(exclude=["number"]).columns),
            )),
            ("model", LogisticRegression(max_iter=2000, random_state=42, class_weight=class_weight)),
        ])
        y_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        rows.append({
            "class_weight": "balanced" if class_weight else "none",
            "precision": precision_score(y_train, y_pred, zero_division=0),
            "recall": recall_score(y_train, y_pred, zero_division=0),
            "f1": f1_score(y_train, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_train, y_proba),
        })

    df = pd.DataFrame(rows)
    print("\nImpacto do class_weight='balanced' no modelo logístico:")
    print(df[["class_weight", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))
    return df


def compare_feature_impact(X_train: pd.DataFrame, y_train: pd.Series, feature_name: str, cv: StratifiedKFold) -> pd.DataFrame:
    rows = []
    for enabled in [True, False]:
        df_subset = X_train.copy()
        if not enabled and feature_name in df_subset.columns:
            df_subset = df_subset.drop(columns=[feature_name])

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(
                list(df_subset.select_dtypes(include=["number"]).columns),
                list(df_subset.select_dtypes(exclude=["number"]).columns),
            )),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ])
        y_proba = cross_val_predict(pipeline, df_subset, y_train, cv=cv, method="predict_proba")[:, 1]
        rows.append({
            "feature": feature_name if enabled else f"sem_{feature_name}",
            "roc_auc": roc_auc_score(y_train, y_proba),
        })

    df = pd.DataFrame(rows)
    print(f"\nImpacto da feature '{feature_name}' na ROC AUC:")
    print(df.to_string(index=False))
    return df


def main() -> None:
    ensure_dataset()

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas.")

    target_candidates = ["inadimplente", "default", "target", "status"]
    target_col = find_column(df, target_candidates)
    if target_col is None:
        raise ValueError("Não foi possível identificar a coluna alvo.")

    df[target_col] = normalize_target(df[target_col])
    if "id_contrato" in df.columns:
        df = df.drop(columns=["id_contrato"])

    df = create_comprometimento_renda(df)

    plot_nulls(df)
    y = df[target_col]
    plot_target_distribution(y)

    score_col = find_column(df, ["score_credito", "score", "pontuacao"])
    finalidade_col = find_column(df, ["finalidade", "motivo_emprestimo"])
    imovel_col = find_column(df, ["posse_imovel", "tem_imovel", "imovel"])

    plot_inadimplencia_vs_score(df, target_col, score_col)
    plot_inadimplencia_por_categoria(df, target_col, finalidade_col, "Inadimplência por finalidade", "eda_inadimplencia_finalidade.png")
    plot_inadimplencia_por_categoria(df, target_col, imovel_col, "Inadimplência por posse de imóvel", "eda_inadimplencia_posse_imovel.png")

    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col].astype(int)

    numeric_cols = list(X.select_dtypes(include=["number"]).columns)
    categorical_cols = list(X.select_dtypes(exclude=["number"]).columns)

    if "comprometimento_renda" not in numeric_cols:
        numeric_cols.append("comprometimento_renda")

    numeric_cols = sorted(set(numeric_cols))
    categorical_cols = sorted(set(categorical_cols))

    print("Colunas numéricas:", numeric_cols)
    print("Colunas categóricas:", categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_df = compare_models_cv(X_train, y_train, cv)
    model_variants = build_model_variants()

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    best_model_name = results_df.iloc[0]["modelo"]
    print(f"\nMelhor modelo inicial: {best_model_name}.")

    best_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model_variants[best_model_name]),
    ])

    param_grid = build_param_grid(best_model_name)
    if param_grid:
        grid = GridSearchCV(
            estimator=best_pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        best_pipeline = grid.best_estimator_
        print(f"Melhores hiperparâmetros: {grid.best_params_}")
        print(f"ROC AUC de validação: {grid.best_score_:.4f}")
    else:
        best_pipeline.fit(X_train, y_train)

    if best_model_name == "LogisticRegression":
        compare_class_weight_effect(X_train, y_train, cv)
    compare_feature_impact(X_train, y_train, "comprometimento_renda", cv)

    cv_train_proba = cross_val_predict(best_pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold_df = evaluate_thresholds_by_cost(y_train.to_numpy(), cv_train_proba)
    threshold = float(threshold_df.iloc[0]["threshold"])

    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_proba, threshold)
    total_cost = threshold_df.iloc[0]["custo_total"]
    plot_roc_curve(y_test.to_numpy(), y_proba, BASE_DIR / "roc_curve.png", title="Curva ROC - Modelo Final")

    print("\nAnálise de limiar por custo no conjunto de treino (cross_val_predict):")
    print(threshold_df.to_string(index=False))
    print(f"\nLimiar recomendado: {threshold:.3f}")
    print(f"Custo total no conjunto de teste com esse limiar: R$ {metrics['fp'] * 1500.0 + metrics['fn'] * 8000.0:,.2f}")
    print(f"Matriz de confusão:\n{metrics['confusion_matrix']}")
    print(f"Precisão: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")

    artifact = {
        "model": best_pipeline,
        "preprocessor": best_pipeline.named_steps["preprocessor"],
        "threshold": threshold,
        "feature_columns": list(X.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": target_col,
        "metrics": metrics,
        "cv_results": results_df.to_dict(orient="records"),
        "best_model_name": best_model_name,
    }

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(artifact, file)

    print(f"\nModelo exportado em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
