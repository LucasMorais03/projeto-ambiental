import os
import glob
import pandas as pd
from datetime import datetime
import requests

# Caminhos
BASE_DIR = os.getcwd()
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

FILENAME_PREFIX = "auto_infracao_2024"  # seu arquivo específico

def find_csv():
    pattern = os.path.join(RAW_DIR, FILENAME_PREFIX + "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Nenhum CSV encontrado começando com {FILENAME_PREFIX} em {RAW_DIR}")
    return files[0]

def load_csv(path):
    attempts = [
        {"sep": ",", "enc": "utf-8"},
        {"sep": ";", "enc": "utf-8"},
        {"sep": ",", "enc": "latin1"},
        {"sep": ";", "enc": "latin1"},
    ]
    for opt in attempts:
        try:
            df = pd.read_csv(path, sep=opt["sep"], encoding=opt["enc"], low_memory=False)
            print(f"Carregado com sep='{opt['sep']}' encoding='{opt['enc']}'")
            return df
        except:
            pass
    # fallback
    df = pd.read_csv(path, engine="python", low_memory=False)
    print("Carregado com fallback padrão.")
    return df

def normalize_columns(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def save_processed(df):
    out = os.path.join(PROCESSED_DIR, f"autuacoes_processed_{datetime.now().strftime('%Y%m%d_%H%M')}.parquet")
    df.to_parquet(out, index=False)
    print("Arquivo salvo:", out)

def main():
    print("🔍 Procurando CSV...")
    csv_path = find_csv()
    print("📁 Arquivo encontrado:", csv_path)

    print("📥 Lendo CSV...")
    df = load_csv(csv_path)

    print("🧹 Normalizando colunas...")
    df = normalize_columns(df)

    print("💾 Salvando parquet processado...")
    save_processed(df)

    print("✅ Ingestão finalizada!")

if __name__ == "__main__":
    main()
