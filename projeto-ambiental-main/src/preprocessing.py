# src/preprocessing.py
import os
import pandas as pd
import numpy as np
from datetime import timedelta

BASE = os.getcwd()
PROC_DIR = os.path.join(BASE, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

def find_latest_parquet():
    files = [os.path.join(PROC_DIR,f) for f in os.listdir(PROC_DIR) if f.startswith("autuacoes_processed") and f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError("Nenhum autuacoes_processed*.parquet em data/processed")
    return sorted(files)[-1]

def parse_valor(x):
    if pd.isna(x): 
        return np.nan
    # tira pontos de milhar e troca vírgula decimal
    try:
        s = str(x).replace(".", "").replace(",", ".")
        return float(s)
    except:
        # tenta extrair números
        import re
        m = re.search(r"(\d+[\.,]?\d*)", str(x))
        if m:
            return float(m.group(1).replace(".", "").replace(",", "."))
        return np.nan

def main():
    p = find_latest_parquet()
    print("Lendo:", p)
    df = pd.read_parquet(p)

    # normaliza colunas (já feito mas garantimos)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # datas: usar 'dat_hora_auto_infracao' ou 'dt_fato_infracional' como referência
    for col in ["dat_hora_auto_infracao", "dt_fato_infracional", "dt_inicio_ato_inequivoco", "dt_fim_ato_inequivoco", "dt_lancamento"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # lat/lon
    lat_cols = ["num_latitude_auto", "num_latitude_auto", "num_latitude_auto"]  # placeholder
    lon_cols = ["num_longitude_auto", "num_longitude_auto", "num_longitude_auto"]
    # detecta nomes corretos
    if "num_latitude_auto" in df.columns and "num_longitude_auto" in df.columns:
        df["lat"] = pd.to_numeric(df["num_latitude_auto"], errors="coerce")
        df["lon"] = pd.to_numeric(df["num_longitude_auto"], errors="coerce")
    elif "num_latitude_auto" in df.columns:
        df["lat"] = pd.to_numeric(df["num_latitude_auto"], errors="coerce")
        df["lon"] = pd.to_numeric(df.get("num_longitude_auto", np.nan), errors="coerce")
    else:
        df["lat"] = np.nan
        df["lon"] = np.nan

    # valor da multa: coluna val_auto_infracao (ex: "15000,00")
    if "val_auto_infracao" in df.columns:
        df["valor_multa"] = df["val_auto_infracao"].apply(parse_valor)
    else:
        df["valor_multa"] = np.nan

    # gravidade: tentar transformar cd_nivel_gravidade em num
    if "cd_nivel_gravidade" in df.columns:
        df["gravidade_nivel"] = pd.to_numeric(df["cd_nivel_gravidade"], errors="coerce")
    else:
        df["gravidade_nivel"] = np.nan

    # coluna de agência/empresa infratora: preferir cpf_cnpj_infrator ou nome_infrator
    if "cpf_cnpj_infrator" in df.columns:
        df["infrator_id"] = df["cpf_cnpj_infrator"].fillna(df.get("num_pessoa_infrator", np.nan)).astype(str)
    else:
        df["infrator_id"] = df.get("num_pessoa_infrator", df.get("nome_infrator", "")).astype(str)

    # campo de município/uf já existem: 'municipio','uf'
    df["municipio"] = df["municipio"].fillna("UNKNOWN")
    df["uf"] = df["uf"].fillna("UNKNOWN")

    # cria coluna year_month para agregações temporais
    if "dat_hora_auto_infracao" in df.columns:
        df["year_month"] = df["dat_hora_auto_infracao"].dt.to_period("M")
    else:
        df["year_month"] = pd.NaT

    # Feature: contagem de autuações por infrator nos últimos 365 dias (rolling)
    if "dat_hora_auto_infracao" in df.columns:
        df = df.sort_values("dat_hora_auto_infracao")
        # para cada infrator contar eventos anteriores 365 dias
        df["autuacoes_365d"] = 0
        # cálculo eficiente por groupby
        def count_last365(sub):
            sub = sub.sort_values("dat_hora_auto_infracao")
            dates = sub["dat_hora_auto_infracao"]
            counts = []
            from bisect import bisect_left
            for i, d in enumerate(dates):
                window_start = d - pd.Timedelta(days=365)
                # número de elementos >= window_start and < d  (exclui corrente)
                j = bisect_left(dates.tolist(), window_start)
                cnt = i - j
                counts.append(cnt)
            sub["autuacoes_365d"] = counts
            return sub
        try:
            df = df.groupby("infrator_id", group_keys=False).apply(count_last365)
        except Exception:
            df["autuacoes_365d"] = 0
    else:
        df["autuacoes_365d"] = 0

    # Agregação por município
    agg = df.groupby(["uf","municipio"], dropna=False).agg(
        qtd_autuacoes = ("seq_auto_infracao","count"),
        soma_multas = ("valor_multa","sum"),
        media_gravidade = ("gravidade_nivel","mean"),
        qtd_com_coord = ("lat","count")
    ).reset_index()

    # salva arquivos
    clean_path = os.path.join(PROC_DIR, "clean_autuacoes.parquet")
    sample_path = os.path.join(PROC_DIR, "sample_for_dashboard.parquet")
    agg_path = os.path.join(PROC_DIR, "agg_municipio.parquet")

    print("Salvando:", clean_path)
    df.to_parquet(clean_path, index=False)
    print("Salvando sample:", sample_path)
    # reduzir colunas para dashboard (evita textos enormes)
    cols_dashboard = ["seq_auto_infracao","dat_hora_auto_infracao","municipio","uf","infrator_id","valor_multa","gravidade_nivel","lat","lon","autuacoes_365d","des_infracao"]
    cols_dashboard = [c for c in cols_dashboard if c in df.columns]
    df[cols_dashboard].head(50000).to_parquet(sample_path, index=False)
    print("Salvando agregação por município:", agg_path)
    agg.to_parquet(agg_path, index=False)

    print("Concluído. Outputs:")
    print(" -", clean_path)
    print(" -", sample_path)
    print(" -", agg_path)

if __name__ == "__main__":
    main()
