# src/model.py
"""
Treina 3 modelos:
 - Classificador (gravidade: baixa/média/alta)  -> RandomForestClassifier
 - Regressor (valor_multa)                     -> RandomForestRegressor
 - Detecção de anomalias                       -> IsolationForest

Entradas:
 - Usa data/processed/clean_autuacoes.parquet como fonte (gerado pelo preprocessing.py)

Saídas (salvas em /models):
 - models/rf_clf.joblib
 - models/rf_reg.joblib
 - models/iso_forest.joblib
 - models/preprocessor.joblib (ColumnTransformer)
 - metrics_summary.txt (resumo das métricas)
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
from math import sqrt

BASE = os.getcwd()
PROC_DIR = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def find_clean():
    candidates = [os.path.join(PROC_DIR,f) for f in os.listdir(PROC_DIR) if f == "clean_autuacoes.parquet"]
    if not candidates:
        # fallback to most recent parquet
        files = [os.path.join(PROC_DIR,f) for f in os.listdir(PROC_DIR) if f.endswith(".parquet")]
        if not files:
            raise FileNotFoundError("Nenhum arquivo parquet em data/processed")
        return sorted(files)[-1]
    return candidates[0]

def build_target_and_features(df):
    df = df.copy()

    if "valor_multa" not in df.columns:
        df["valor_multa"] = np.nan

    if "gravidade_nivel" in df.columns and df["gravidade_nivel"].notna().sum() > 100:
        y_raw = pd.to_numeric(df["gravidade_nivel"], errors="coerce")
        if y_raw.notna().sum() < 50:
            y_raw = None
    else:
        y_raw = None

    if y_raw is None:
        q1 = df["valor_multa"].quantile(0.25)
        q3 = df["valor_multa"].quantile(0.75)
        def proxy(v):
            try:
                if np.isnan(v):
                    return 1
                if v < q1:
                    return 0
                if v >= q3:
                    return 2
                return 1
            except:
                return 1
        y_cls = df["valor_multa"].apply(proxy).astype(int)
    else:
        q1 = y_raw.quantile(0.25)
        q3 = y_raw.quantile(0.75)
        def mapg(x):
            if np.isnan(x): return 1
            if x < q1: return 0
            if x >= q3: return 2
            return 1
        y_cls = y_raw.apply(mapg).astype(int)

    y_reg = df["valor_multa"].fillna(0.0).astype(float)

    features = []

    if "dat_hora_auto_infracao" in df.columns:
        df["year"] = pd.to_datetime(df["dat_hora_auto_infracao"], errors="coerce").dt.year
        df["month"] = pd.to_datetime(df["dat_hora_auto_infracao"], errors="coerce").dt.month
        features += ["year", "month"]

    if "uf" in df.columns:
        features.append("uf")
    if "municipio" in df.columns:
        features.append("municipio")

    if "autuacoes_365d" in df.columns:
        df["autuacoes_365d"] = pd.to_numeric(df["autuacoes_365d"], errors="coerce").fillna(0)
        features.append("autuacoes_365d")

    if "gravidade_nivel" in df.columns:
        df["gravidade_nivel"] = pd.to_numeric(df["gravidade_nivel"], errors="coerce").fillna(-1)
        features.append("gravidade_nivel")

    if "lat" in df.columns and "lon" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce").fillna(0)
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce").fillna(0)
        features += ["lat","lon"]

    if "des_infracao" in df.columns:
        df["desc_len"] = df["des_infracao"].astype(str).apply(len)
        features.append("desc_len")

    for c in features:
        if c not in df.columns:
            df[c] = 0

    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, y_cls, y_reg, df

def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)

        ],
        remainder="drop",
        sparse_threshold=0
    )
    return transformer, numeric_cols, cat_cols

def train_models():
    path = find_clean()
    print("Lendo dados:", path)
    df = pd.read_parquet(path)
    X, y_cls, y_reg, df_full = build_target_and_features(df)

    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_t, y_cls_train)
    preds_cls = clf.predict(X_test_t)
    report = classification_report(y_cls_test, preds_cls, digits=3)
    print(report)
    joblib.dump(clf, os.path.join(MODEL_DIR, "rf_clf.joblib"))

    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg.fit(X_train_t, y_reg_train)
    preds_reg = reg.predict(X_test_t)
    mae = mean_absolute_error(y_reg_test, preds_reg)
    rmse = sqrt(mean_squared_error(y_reg_test, preds_reg))
    print("MAE:", mae, "RMSE:", rmse)
    joblib.dump(reg, os.path.join(MODEL_DIR, "rf_reg.joblib"))

    iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=42)
    iso.fit(np.vstack([X_train_t, X_test_t]))
    joblib.dump(iso, os.path.join(MODEL_DIR, "iso_forest.joblib"))

    with open(os.path.join(MODEL_DIR, "metrics_summary.txt"), "w") as f:
        f.write(report)
        f.write(f"\nMAE: {mae}\nRMSE: {rmse}\n")

    print("Modelos treinados e salvos em", MODEL_DIR)

if __name__ == "__main__":
    train_models()
